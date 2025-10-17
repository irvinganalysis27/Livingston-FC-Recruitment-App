# lib/scoring.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Set

import numpy as np
import pandas as pd
import re


# ============================================================
# Constants & mappings (same as Team Rankings page)
# ============================================================
RAW_TO_GROUP = {
    "LEFTBACK": "Full Back", "LEFTWINGBACK": "Full Back",
    "RIGHTBACK": "Full Back", "RIGHTWINGBACK": "Full Back",
    "CENTREBACK": "Centre Back", "LEFTCENTREBACK": "Centre Back", "RIGHTCENTREBACK": "Centre Back",
    "DEFENSIVEMIDFIELDER": "Number 6", "LEFTDEFENSIVEMIDFIELDER": "Number 6",
    "RIGHTDEFENSIVEMIDFIELDER": "Number 6", "CENTREDEFENSIVEMIDFIELDER": "Number 6",
    "CENTREMIDFIELDER": "Number 8", "LEFTCENTREMIDFIELDER": "Number 8", "RIGHTCENTREMIDFIELDER": "Number 8",
    "CENTREATTACKINGMIDFIELDER": "Number 8", "RIGHTATTACKINGMIDFIELDER": "Number 8",
    "LEFTATTACKINGMIDFIELDER": "Number 8", "SECONDSTRIKER": "Number 8",
    "LEFTWING": "Winger", "LEFTMIDFIELDER": "Winger",
    "RIGHTWING": "Winger", "RIGHTMIDFIELDER": "Winger",
    "CENTREFORWARD": "Striker", "LEFTCENTREFORWARD": "Striker", "RIGHTCENTREFORWARD": "Striker",
    "GOALKEEPER": "Goalkeeper",
}

DEFAULT_LOWER_IS_BETTER: Set[str] = {"Turnovers", "Fouls", "Pr. Long Balls", "UPr. Long Balls"}

POS_COL = "Six-Group Position"
MINUTES_COL = "Minutes played"
COMPETITION_NORM = "Competition_norm"
MULTIPLIER_COL = "Multiplier"
LFC_MULTIPLIER_COL = "LFC Multiplier"

# ============================================================
# Small helpers
# ============================================================
def _clean_pos_token(tok: str) -> str:
    if pd.isna(tok): 
        return ""
    t = str(tok).upper().strip()
    t = re.sub(r"[.\-_/]", " ", t)
    return re.sub(r"\s+", "", t)

def map_first_position_to_group(primary_pos_cell) -> Optional[str]:
    return RAW_TO_GROUP.get(_clean_pos_token(primary_pos_cell), None)

def _to100(v, lo, hi) -> float:
    """Min-max to 0–100 with safe fallbacks (same behaviour as page)."""
    try:
        v = float(v)
        lo = float(lo)
        hi = float(hi)
    except (TypeError, ValueError):
        return 50.0  # neutral
    if not np.isfinite(v) or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 50.0
    return float(np.clip((v - lo) / (hi - lo) * 100.0, 0.0, 100.0))


# ============================================================
# Preprocess: light normalisation + multipliers + positions
# ============================================================
@dataclass
class PreprocessConfig:
    root_dir: Path
    league_multiplier_filename: str = "league_multipliers.xlsx"
    # If Competition_norm is missing, copy from Competition:
    ensure_comp_norm_from: Optional[str] = "Competition"
    # Columns to standardise:
    rename_name_to_player: bool = True
    rename_primary_pos_to_position: bool = True
    rename_minutes_to_minutes_played: bool = True
    # Build "Positions played" from Position [+ Secondary Position]:
    build_positions_played: bool = True
    # Create Six-Group Position via RAW_TO_GROUP:
    create_six_group_pos: bool = True


def preprocess_for_scoring(
    df_in: pd.DataFrame,
    cfg: PreprocessConfig
) -> pd.DataFrame:
    """
    Mirrors the Team Rankings preprocessing used for scoring:
    - renames key columns,
    - ensures Competition_norm,
    - merges league multipliers,
    - builds 'Positions played',
    - maps to 'Six-Group Position'.
    """
    df = df_in.copy()

    # ---- Rename identifiers to match the scoring code expectations
    rename_map = {}
    if cfg.rename_name_to_player and "Name" in df.columns:
        rename_map["Name"] = "Player"
    if cfg.rename_primary_pos_to_position and "Primary Position" in df.columns:
        rename_map["Primary Position"] = "Position"
    if cfg.rename_minutes_to_minutes_played and "Minutes" in df.columns:
        rename_map["Minutes"] = MINUTES_COL
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # ---- Ensure Competition_norm exists
    if COMPETITION_NORM not in df.columns and cfg.ensure_comp_norm_from and cfg.ensure_comp_norm_from in df.columns:
        df[COMPETITION_NORM] = df[cfg.ensure_comp_norm_from].astype(str)

    # ---- Merge league multipliers
    try:
        mult_path = cfg.root_dir / cfg.league_multiplier_filename
        mult = pd.read_excel(mult_path)
        if {"League", "Multiplier"}.issubset(mult.columns):
            df = df.merge(mult, left_on=COMPETITION_NORM, right_on="League", how="left")
            df[MULTIPLIER_COL] = pd.to_numeric(df.get(MULTIPLIER_COL, np.nan), errors="coerce").fillna(1.0)
        else:
            df[MULTIPLIER_COL] = 1.0
    except Exception:
        df[MULTIPLIER_COL] = 1.0

    # ---- Build "Positions played"
    if cfg.build_positions_played:
        if "Secondary Position" in df.columns and "Position" in df.columns:
            df["Positions played"] = df["Position"].fillna("") + np.where(
                df["Secondary Position"].notna(), ", " + df["Secondary Position"].astype(str), ""
            )
        elif "Position" in df.columns:
            df["Positions played"] = df["Position"]
        else:
            df["Positions played"] = np.nan

    # ---- Map to six groups
    if cfg.create_six_group_pos:
        if "Position" in df.columns:
            df[POS_COL] = df["Position"].apply(map_first_position_to_group)
        else:
            df[POS_COL] = np.nan

    return df


# ============================================================
# Scoring (Team Rankings page logic)
# ============================================================
@dataclass
class ScoringConfig:
    min_minutes_for_baseline: int = 600
    lower_is_better: Iterable[str] = tuple(DEFAULT_LOWER_IS_BETTER)
    pos_col: str = POS_COL
    minutes_col: str = MINUTES_COL
    multiplier_col: str = MULTIPLIER_COL
    # LFC variant:
    lfc_enable: bool = True
    lfc_league_name: str = "Scotland Premiership"
    lfc_multiplier_value: float = 1.20
    # Output column names:
    avg_z_col: str = "Avg Z Score"
    weighted_z_col: str = "Weighted Z Score"
    lfc_weighted_z_col: str = "LFC Weighted Z"
    score_100_col: str = "Score (0–100)"
    score_100_lfc_col: str = "LFC Score (0–100)"


def compute_scores(
    df_all_in: pd.DataFrame,
    cfg: ScoringConfig = ScoringConfig()
) -> pd.DataFrame:
    """
    Compute Z, Weighted Z, and 0–100 scores using the same logic as Team Rankings.
    - Baseline for mean/std is players with minutes >= cfg.min_minutes_for_baseline (fallback: whole df).
    - Z per metric per position group.
    - Weighted Z: positive Z * multiplier; negative Z / multiplier.
    - LFC variant scales with special multiplier for a specific league.
    - Per-position anchors (min/max on Weighted Z among eligible) → 0–100 scale.
    """
    df_all = df_all_in.copy()

    # Ensure needed columns exist
    if cfg.pos_col not in df_all.columns:
        df_all[cfg.pos_col] = np.nan
    if cfg.minutes_col not in df_all.columns:
        df_all[cfg.minutes_col] = np.nan
    if cfg.multiplier_col not in df_all.columns:
        df_all[cfg.multiplier_col] = 1.0

    # ---------- Eligible baseline set ----------
    mins = pd.to_numeric(df_all.get(cfg.minutes_col, np.nan), errors="coerce").fillna(0)
    eligible = df_all[mins >= cfg.min_minutes_for_baseline].copy()
    if eligible.empty:
        eligible = df_all.copy()

    # ---------- Compute Z per numeric column per position ----------
    num_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()

    # Do not compute Z on these control columns:
    exclude = {cfg.minutes_col, "Age", "Height", cfg.multiplier_col}
    metric_cols = [c for c in num_cols if c not in exclude]

    # Pre-calc baseline mean/std using eligible only
    baseline_stats = eligible.groupby(cfg.pos_col)[metric_cols].agg(["mean", "std"]).fillna(0)
    baseline_stats.columns = baseline_stats.columns.map("_".join)  # e.g., NP Goals_mean

    raw_z = pd.DataFrame(index=df_all.index, columns=metric_cols, dtype=float)

    for m in metric_cols:
        mean_col = f"{m}_mean"
        std_col  = f"{m}_std"
        if mean_col not in baseline_stats.columns or std_col not in baseline_stats.columns:
            continue

        mean_vals = df_all[cfg.pos_col].map(baseline_stats[mean_col])
        # Avoid divide-by-zero by replacing 0 with 1
        std_vals  = df_all[cfg.pos_col].map(baseline_stats[std_col].replace(0, 1))
        z = (pd.to_numeric(df_all[m], errors="coerce") - mean_vals) / std_vals
        if m in set(cfg.lower_is_better):
            z *= -1
        raw_z[m] = z.fillna(0)

    # Average Z across metrics
    df_all[cfg.avg_z_col] = raw_z.mean(axis=1).fillna(0)

    # ---------- Weighted Z ----------
    mult = pd.to_numeric(df_all.get(cfg.multiplier_col, 1.0), errors="coerce").fillna(1.0)
    avg_z = pd.to_numeric(df_all[cfg.avg_z_col], errors="coerce").fillna(0.0)

    df_all[cfg.weighted_z_col] = np.select(
        [avg_z > 0, avg_z < 0],
        [avg_z * mult, avg_z / mult],
        default=0.0
    )

    # ---------- LFC variant (optional) ----------
    if cfg.lfc_enable:
        # Start from 'normal' multipliers
        lfc_mult_series = mult.copy()
        if COMPETITION_NORM in df_all.columns:
            mask = (df_all[COMPETITION_NORM] == cfg.lfc_league_name)
            lfc_mult_series = np.where(mask, float(cfg.lfc_multiplier_value), lfc_mult_series)

        df_all[LFC_MULTIPLIER_COL] = lfc_mult_series

        df_all[cfg.lfc_weighted_z_col] = np.select(
            [avg_z > 0, avg_z < 0],
            [avg_z * lfc_mult_series, avg_z / lfc_mult_series],
            default=0.0
        )

    # ---------- Anchors from eligible (per position, on Weighted Z) ----------
    eligible_for_anchors = df_all[mins >= cfg.min_minutes_for_baseline].copy()
    if eligible_for_anchors.empty:
        eligible_for_anchors = df_all.copy()

    anchors = (
        eligible_for_anchors
        .groupby(cfg.pos_col, dropna=False)[cfg.weighted_z_col]
        .agg(_scale_min="min", _scale_max="max")
        .fillna(0)
    )

    if anchors.empty:
        df_all["_scale_min"] = 0.0
        df_all["_scale_max"] = 1.0
    else:
        df_all = df_all.merge(anchors, left_on=cfg.pos_col, right_index=True, how="left")

    # ---------- Scale to 0–100 ----------
    df_all[cfg.score_100_col] = [
        _to100(v, lo, hi)
        for v, lo, hi in zip(df_all[cfg.weighted_z_col], df_all["_scale_min"], df_all["_scale_max"])
    ]

    if cfg.lfc_enable and cfg.lfc_weighted_z_col in df_all.columns:
        df_all[cfg.score_100_lfc_col] = [
            _to100(v, lo, hi)
            for v, lo, hi in zip(df_all[cfg.lfc_weighted_z_col], df_all["_scale_min"], df_all["_scale_max"])
        ]

    # Final tidy
    df_all[cfg.score_100_col] = pd.to_numeric(df_all[cfg.score_100_col], errors="coerce").round(1).fillna(0)
    if cfg.lfc_enable and cfg.score_100_lfc_col in df_all.columns:
        df_all[cfg.score_100_lfc_col] = (
            pd.to_numeric(df_all[cfg.score_100_lfc_col], errors="coerce").round(1).fillna(0)
        )

    df_all.drop(columns=["_scale_min", "_scale_max"], inplace=True, errors="ignore")
    return df_all


# ============================================================
# Convenience one-liners
# ============================================================
def score_pipeline(
    df_in: pd.DataFrame,
    root_dir: Path,
    *,
    preprocess_config: Optional[PreprocessConfig] = None,
    scoring_config: Optional[ScoringConfig] = None,
) -> pd.DataFrame:
    """
    Full pipeline used by Team Rankings:
      preprocess_for_scoring(...) ➜ compute_scores(...)
    """
    if preprocess_config is None:
        preprocess_config = PreprocessConfig(root_dir=root_dir)
    if scoring_config is None:
        scoring_config = ScoringConfig()

    df_prep = preprocess_for_scoring(df_in, preprocess_config)
    df_scored = compute_scores(df_prep, scoring_config)
    return df_scored


__all__ = [
    # constants
    "POS_COL", "MINUTES_COL", "COMPETITION_NORM", "MULTIPLIER_COL",
    "DEFAULT_LOWER_IS_BETTER",
    # configs
    "PreprocessConfig", "ScoringConfig",
    # funcs
    "map_first_position_to_group",
    "preprocess_for_scoring",
    "compute_scores",
    "score_pipeline",
]
