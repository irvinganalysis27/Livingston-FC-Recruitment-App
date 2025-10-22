import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# ⚙️ CONFIG CONSTANTS
# ============================================================
LOWER_IS_BETTER = {"Turnovers", "Fouls", "Pr. Long Balls", "UPr. Long Balls"}

# ============================================================
# 🧩 Percentile Helper
# ============================================================
def pct_rank(series: pd.Series, lower_is_better: bool = False) -> pd.Series:
    """Return percentile ranks from 0–100. Inverts scale for lower-is-better metrics."""
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    r = s.rank(pct=True, ascending=True)
    if lower_is_better:
        p = 1.0 - r
    else:
        p = r
    return (p * 100.0).round(1)

# ============================================================
# 🧠 Preprocess for scoring
# ============================================================
def preprocess_for_scoring(df: pd.DataFrame, metrics: list, position_col: str) -> pd.DataFrame:
    """
    Ensure required metrics exist, numeric, and fill NaNs.
    Adds basic structure before computing percentiles & scores.
    """
    df = df.copy()
    for m in metrics:
        if m not in df.columns:
            df[m] = 0
        df[m] = pd.to_numeric(df[m], errors="coerce").fillna(0)
    if position_col not in df.columns:
        df[position_col] = np.nan
    return df

# ============================================================
# 🧮 Compute all scores (identical to app logic)
# ============================================================
def compute_scores(df: pd.DataFrame,
                   metrics: list,
                   position_col: str = "Six-Group Position",
                   minutes_col: str = "Minutes played",
                   league_col: str = "Competition_norm",
                   multiplier_col: str = "Multiplier",
                   within_league: bool = True) -> pd.DataFrame:
    """
    Compute percentiles, Z-scores, Weighted Z, LFC Weighted Z, and 0–100 scores.
    Returns a new dataframe including all derived columns.
    """
    df = df.copy()

    # ============================================================
    # A) Percentiles for radar chart (within league or global)
    # ============================================================
    percentile_df_chart = pd.DataFrame(index=df.index, columns=metrics, dtype=float)

    if within_league and league_col in df.columns:
        for m in metrics:
            try:
                percentile_df_chart[m] = (
                    df.groupby(league_col, group_keys=False)[m]
                      .apply(lambda s: pct_rank(s, lower_is_better=(m in LOWER_IS_BETTER)))
                )
            except Exception:
                percentile_df_chart[m] = 50.0
    else:
        for m in metrics:
            try:
                percentile_df_chart[m] = pct_rank(df[m], lower_is_better=(m in LOWER_IS_BETTER))
            except Exception:
                percentile_df_chart[m] = 50.0

    percentile_df_chart = percentile_df_chart.fillna(50.0).round(1)

    # ============================================================
    # B) Global Percentiles (by position, baseline for scoring)
    # ============================================================
    percentile_df_global = pd.DataFrame(index=df.index, columns=metrics, dtype=float)
    for m in metrics:
        try:
            percentile_df_global[m] = (
                df.groupby(position_col, group_keys=False)[m]
                  .apply(lambda s: pct_rank(s, lower_is_better=(m in LOWER_IS_BETTER)))
            )
        except Exception:
            percentile_df_global[m] = 50.0
    percentile_df_global = percentile_df_global.fillna(50.0).round(1)

    # ============================================================
    # C) Raw Z-Scores (per position)
    # ============================================================
    raw_z_all = pd.DataFrame(index=df.index, columns=metrics, dtype=float)
    for m in metrics:
        z_per_group = df.groupby(position_col)[m].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
        )
        if m in LOWER_IS_BETTER:
            z_per_group *= -1
        raw_z_all[m] = z_per_group.fillna(0)

    df["Avg Z Score"] = raw_z_all.mean(axis=1).fillna(0)

    # ============================================================
    # D) Weighted Z (league multipliers)
    # ============================================================
    mult = pd.to_numeric(df.get(multiplier_col, 1.0), errors="coerce").fillna(1.0)
    avg_z = df["Avg Z Score"]

    df["Weighted Z Score"] = np.select(
        [avg_z > 0, avg_z < 0],
        [avg_z * mult, avg_z / mult],
        default=0.0
    )

    # ============================================================
    # E) LFC Weighted Z (Scottish Premiership 1.20)
    # ============================================================
    df["LFC Multiplier"] = mult
    df.loc[df[league_col] == "Scotland Premiership", "LFC Multiplier"] = 1.20
    lfc_mult = df["LFC Multiplier"]

    df["LFC Weighted Z"] = np.select(
        [avg_z > 0, avg_z < 0],
        [avg_z * lfc_mult, avg_z / lfc_mult],
        default=0.0
    )

    # ============================================================
    # F) Convert to 0–100 score (anchors per position)
    # ============================================================
    mins_num = pd.to_numeric(df.get(minutes_col, np.nan), errors="coerce")
    eligible = df[mins_num >= 600].copy()
    if eligible.empty:
        eligible = df.copy()

    anchors = (
        eligible.groupby(position_col, dropna=False)["Weighted Z Score"]
        .agg(_scale_min="min", _scale_max="max")
        .fillna(0)
    )
    df = df.merge(anchors, left_on=position_col, right_index=True, how="left")

    def _to100(v, lo, hi):
        if pd.isna(v) or pd.isna(lo) or pd.isna(hi) or hi <= lo:
            return 50.0
        return np.clip((v - lo) / (hi - lo) * 100.0, 0.0, 100.0)

    df["Score (0–100)"] = [
        _to100(v, lo, hi)
        for v, lo, hi in zip(df["Weighted Z Score"], df["_scale_min"], df["_scale_max"])
    ]
    df["LFC Score (0–100)"] = [
        _to100(v, lo, hi)
        for v, lo, hi in zip(df["LFC Weighted Z"], df["_scale_min"], df["_scale_max"])
    ]

    df[["Score (0–100)", "LFC Score (0–100)"]] = (
        df[["Score (0–100)", "LFC Score (0–100)"]]
        .apply(pd.to_numeric, errors="coerce")
        .round(1)
        .fillna(0)
    )

    # ============================================================
    # G) Return all new columns together
    # ============================================================
    result = df.copy()
    for m in metrics:
        pct_col = f"{m} (percentile)"
        result[pct_col] = percentile_df_chart[m]
    return result
