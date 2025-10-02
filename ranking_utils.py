# ranking_utils.py

import pandas as pd
import numpy as np

# ---------- Position metrics ----------
position_metrics = {
    "Centre Back": {
        "metrics": [
            "NP Goals",
            "Passing%", "Pass OBV", "Pr. Long Balls", "UPr. Long Balls", "OBV", "Pr. Pass% Dif.",
            "PAdj Interceptions", "PAdj Tackles", "Dribbles Stopped%",
            "Defensive Actions", "Aggressive Actions", "Fouls",
            "Aerial Wins", "Aerial Win%",
        ],
        "groups": {
            "PAdj Interceptions": "Defensive",
            "PAdj Tackles": "Defensive",
            "Dribbles Stopped%": "Defensive",
            "Defensive Actions": "Defensive",
            "Aggressive Actions": "Defensive",
            "Fouls": "Defensive",
            "Aerial Wins": "Defensive",
            "Aerial Win%": "Defensive",
            "Passing%": "Possession",
            "Pr. Pass% Dif.": "Possession",
            "Pr. Long Balls": "Possession",
            "UPr. Long Balls": "Possession",
            "OBV": "Possession",
            "Pass OBV": "Possession",
            "NP Goals": "Attacking",
        }
    },
    "Full Back": {
        "metrics": [
            "Passing%", "Pr. Pass% Dif.", "Successful Crosses", "Crossing%", "Deep Progressions",
            "Successful Dribbles", "Turnovers", "OBV", "Pass OBV",
            "Defensive Actions", "Aerial Win%", "PAdj Pressures",
            "PAdj Tack&Int", "Dribbles Stopped%", "Aggressive Actions", "Player Season Ball Recoveries 90"
        ],
        "groups": {}  # same structure as radar page
    },
    "Number 6": {
        "metrics": [
            "xGBuildup", "xG Assisted",
            "Passing%", "Deep Progressions", "Turnovers", "OBV", "Pass OBV", "Pr. Pass% Dif.",
            "PAdj Interceptions", "PAdj Tackles", "Dribbles Stopped%",
            "Aggressive Actions", "Aerial Win%", "Player Season Ball Recoveries 90", "Pressure Regains",
        ],
        "groups": {}
    },
    "Number 8": {
        "metrics": [
            "xGBuildup", "xG Assisted", "Shots", "xG", "NP Goals",
            "Passing%", "Deep Progressions", "OP Passes Into Box", "Pass OBV", "OBV", "Deep Completions",
            "Pressure Regains", "PAdj Pressures", "Player Season Fhalf Ball Recoveries 90",
            "Aggressive Actions",
        ],
        "groups": {}
    },
    "Winger": {
        "metrics": [
            "xG", "Shots", "xG/Shot", "Touches In Box", "OP xG Assisted", "NP Goals",
            "OP Passes Into Box", "Successful Box Cross%", "Passing%",
            "Successful Dribbles", "Turnovers", "OBV", "D&C OBV", "Fouls Won", "Deep Progressions",
            "Player Season Fhalf Pressures 90",
        ],
        "groups": {}
    },
    "Striker": {
        "metrics": [
            "Aggressive Actions", "NP Goals", "xG", "Shots", "xG/Shot",
            "Goal Conversion%",
            "Touches In Box", "xG Assisted",
            "Fouls Won", "Deep Completions", "OP Key Passes",
            "Aerial Win%", "Aerial Wins", "Player Season Fhalf Pressures 90",
        ],
        "groups": {}
    },
    "Goalkeeper": {
        "metrics": [
            "Goals Conceded", "PSxG Faced", "GSAA", "Save%", "xSv%", "Shot Stopping%",
            "Shots Faced", "Shots Faced OT%", "Positive Outcome%", "Goalkeeper OBV",
        ],
        "groups": {}
    }
}

# ---------- Lower-is-better ----------
LOWER_IS_BETTER = {
    "Turnovers", "Fouls", "Pr. Long Balls", "UPr. Long Balls"
}

# ---------- Helpers ----------
def pct_rank(series: pd.Series, lower_is_better: bool) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").fillna(0)
    r = series.rank(pct=True, ascending=True)
    return ((1.0 - r) if lower_is_better else r) * 100.0

def _minmax_score(val, lo, hi):
    try:
        val, lo, hi = float(val), float(lo), float(hi)
    except (TypeError, ValueError):
        return 0.0
    if hi <= lo: return 50.0
    return np.clip((val - lo) / (hi - lo) * 100.0, 0.0, 100.0)

# ---------- Main ----------
def compute_rankings(df_all: pd.DataFrame, min_minutes: int = 600) -> pd.DataFrame:
    pos_col = "Six-Group Position"
    minutes_col = "Minutes played"

    if pos_col not in df_all.columns: df_all[pos_col] = np.nan
    if minutes_col not in df_all.columns: df_all[minutes_col] = np.nan
    if "Multiplier" not in df_all.columns: df_all["Multiplier"] = 1.0

    # All metrics
    all_metrics = {m for role in position_metrics.values() for m in role["metrics"]}
    df_all = df_all.copy()
    for m in all_metrics:
        if m not in df_all.columns:
            df_all[m] = 0
        df_all[m] = pd.to_numeric(df_all[m], errors="coerce").fillna(0)

    # Z-scores
    raw_z_all = pd.DataFrame(index=df_all.index, columns=all_metrics, dtype=float)
    for m in all_metrics:
        z = df_all.groupby(pos_col)[m].transform(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0)
        if m in LOWER_IS_BETTER:
            z *= -1
        raw_z_all[m] = z.fillna(0)

    df_all["Avg Z Score"] = raw_z_all.mean(axis=1)

    # Weighted
    df_all["Multiplier"] = pd.to_numeric(df_all["Multiplier"], errors="coerce").fillna(1.0)
    df_all["Weighted Z Score"] = df_all["Avg Z Score"] * df_all["Multiplier"]

    # Anchors
    mins = pd.to_numeric(df_all[minutes_col], errors="coerce").fillna(0)
    eligible = df_all[mins >= min_minutes].copy()
    if eligible.empty: eligible = df_all.copy()

    anchor_minmax = (
        eligible.groupby(pos_col)["Weighted Z Score"]
                .agg(_scale_min="min", _scale_max="max")
                .fillna(0)
    )
    df_all = df_all.merge(anchor_minmax, left_on=pos_col, right_index=True, how="left")

    # Scale
    df_all["Score (0–100)"] = [
        _minmax_score(v, lo, hi)
        for v, lo, hi in zip(df_all["Weighted Z Score"], df_all["_scale_min"], df_all["_scale_max"])
    ]
    df_all["Score (0–100)"] = pd.to_numeric(df_all["Score (0–100)"], errors="coerce").round(1).fillna(0)

    # Rank
    df_all["Rank"] = (
        df_all.groupby(pos_col)["Score (0–100)"]
              .rank(ascending=False, method="min")
    )

    return df_all
