# lib/scoring.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Set, List

import numpy as np
import pandas as pd
import re

# ============================================================
# Constants & mappings (aligned with pages)
# ============================================================
RAW_TO_GROUP = {
    # Full backs & wing backs
    "RIGHTBACK": "Full Back", "LEFTBACK": "Full Back",
    "RIGHTWINGBACK": "Full Back", "LEFTWINGBACK": "Full Back",
    # Centre backs
    "RIGHTCENTREBACK": "Centre Back", "LEFTCENTREBACK": "Centre Back", "CENTREBACK": "Centre Back",
    # Sixes
    "DEFENSIVEMIDFIELDER": "Number 6", "RIGHTDEFENSIVEMIDFIELDER": "Number 6",
    "LEFTDEFENSIVEMIDFIELDER": "Number 6", "CENTREDEFENSIVEMIDFIELDER": "Number 6",
    # Generic CM → will optionally duplicate into 6 & 8 outside scoring if desired
    "CENTREMIDFIELDER": "Number 8", "LEFTCENTREMIDFIELDER": "Number 8", "RIGHTCENTREMIDFIELDER": "Number 8",
    # Eights / 10s
    "CENTREATTACKINGMIDFIELDER": "Number 8", "ATTACKINGMIDFIELDER": "Number 8",
    "RIGHTATTACKINGMIDFIELDER": "Number 8", "LEFTATTACKINGMIDFIELDER": "Number 8",
    "SECONDSTRIKER": "Number 8", "10": "Number 8",
    # Wingers
    "RIGHTWING": "Winger", "LEFTWING": "Winger",
    "RIGHTMIDFIELDER": "Winger", "LEFTMIDFIELDER": "Winger",
    # Strikers
    "CENTREFORWARD": "Striker", "RIGHTCENTREFORWARD": "Striker", "LEFTCENTREFORWARD": "Striker",
    # GK (usually excluded downstream)
    "GOALKEEPER": "Goalkeeper",
}

DEFAULT_LOWER_IS_BETTER: Set[str] = {"Turnovers", "Fouls", "Pr. Long Balls", "UPr. Long Balls"}

POS_COL = "Six-Group Position"
MINUTES_COL = "Minutes played"
COMPETITION_COL = "Competition"
COMPETITION_NORM = "Competition_norm"
MULTIPLIER_COL = "Multiplier"
LFC_MULTIPLIER_COL = "LFC Multiplier"

# ============================================================
# Helpers
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
    """Safe min-max to 0–100."""
    try:
        v = float(v); lo = float(lo); hi = float(hi)
    except (TypeError, ValueError):
        return 50.0
    if not np.isfinite(v) or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 50.0
    return float(np.clip((v - lo) / (hi - lo) * 100.0, 0.0, 100.0))

def _lc(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

# ============================================================
# Preprocess config
# ============================================================
@dataclass
class PreprocessConfig:
    root_dir: Path
    league_multiplier_filename: str = "league_multipliers.xlsx"
    # Build/ensure normed competition
    ensure_comp_norm_from: Optional[str] = COMPETITION_COL
    # Column renames
    rename_name_to_player: bool = True
    rename_primary_pos_to_position: bool = True
    rename_minutes_to_minutes_played: bool = True
    # Build "Positions played"
    build_positions_played: bool = True
    # Map to six groups
    create_six_group_pos: bool = True
    # Apply metric synonyms/renames
    apply_metric_synonyms: bool = True
    # Prefer competition_id for multiplier merge when available
    use_competition_id_when_present: bool = True

# ============================================================
# Scoring config
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

# ============================================================
# Preprocess
# ============================================================
def preprocess_for_scoring(df_in: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    """
    Light, robust preprocessing:
      - ID renames (Name→Player, etc.)
      - Competition_norm creation
      - League multipliers merge (competition_id > normalized league)
      - Metric synonyms/renames (to match radar/team metric names)
      - Position strings and Six-Group mapping
    """
    df = df_in.copy()

    # ---- ID renames
    rename_map = {}
    if cfg.rename_name_to_player and "Name" in df.columns:
        rename_map["Name"] = "Player"
    if cfg.rename_primary_pos_to_position and "Primary Position" in df.columns:
        rename_map["Primary Position"] = "Position"
    if cfg.rename_minutes_to_minutes_played and "Minutes" in df.columns:
        rename_map["Minutes"] = MINUTES_COL
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # ---- Ensure Competition_norm (string)
    if COMPETITION_NORM not in df.columns and cfg.ensure_comp_norm_from and cfg.ensure_comp_norm_from in df.columns:
        df[COMPETITION_NORM] = df[cfg.ensure_comp_norm_from].astype(str)
    if COMPETITION_NORM in df.columns:
        df[COMPETITION_NORM] = df[COMPETITION_NORM].astype(str).str.strip()

    # ---- Metric synonyms (so radar metric names line up)
    if cfg.apply_metric_synonyms:
        # Strict direct renames first
        synonyms = {
            "Successful Box Cross %": "Successful Box Cross%",
            "Player Season Box Cross Ratio": "Successful Box Cross%",
            "Player Season Change In Passing Ratio": "Pr. Pass% Dif.",
            "Player Season Xgbuildup 90": "xGBuildup",
            "Player Season F3 Pressures 90": "Pressures in Final 1/3",
            "Player Season Pressured Long Balls 90": "Pr. Long Balls",
            "Player Season Unpressured Long Balls 90": "UPr. Long Balls",
        }
        df.rename(columns={k: v for k, v in synonyms.items() if k in df.columns}, inplace=True)

        # Pattern-based normalisation to handle small header variations
        def _normalise_metric_names(cols: List[str]) -> List[str]:
            out = []
            for c in cols:
                key = c.strip().replace(" ", "").lower()
                if key in {"playerseasonxgassisted90", "playerseasonopxgassisted90"}:
                    out.append("xG Assisted")
                elif key in {"playerseasonoppassesintobox90", "oppassesintobox"}:
                    out.append("OP Passes Into Box")
                elif key in {"playerseasonxgbuildup90", "xgbuildup"}:
                    out.append("xGBuildup")
                elif key in {"playerseasonobv90", "obv"}:
                    out.append("OBV")
                elif key in {"playerseasondcobv90", "dcobv"}:
                    out.append("D&C OBV")
                elif key in {"playerseasonpressuredlongballs90", "prlongballs"}:
                    out.append("Pr. Long Balls")
                elif key in {"playerseasonunpressuredlongballs90", "uprlongballs"}:
                    out.append("UPr. Long Balls")
                else:
                    out.append(c)
            return out

        df.columns = _normalise_metric_names(list(df.columns))

        # Derived examples we used previously
        if "Player Season Total Dribbles 90" in df.columns and "Player Season Failed Dribbles 90" in df.columns:
            df["Successful Dribbles"] = (
                pd.to_numeric(df["Player Season Total Dribbles 90"], errors="coerce").fillna(0)
                - pd.to_numeric(df["Player Season Failed Dribbles 90"], errors="coerce").fillna(0)
            )

        # A generic Successful Crosses derivation when raw counts/ratios exist (best-effort)
        cross_cols = [c for c in df.columns if "crosses" in c.lower()]
        crossperc_cols = [c for c in df.columns if "crossing%" in c.lower() or "successful box cross%" in c.lower()]
        if cross_cols and crossperc_cols:
            try:
                df["Successful Crosses"] = (
                    pd.to_numeric(df[cross_cols[0]], errors="coerce")
                    * (pd.to_numeric(df[crossperc_cols[0]], errors="coerce") / 100.0)
                )
            except Exception:
                pass

    # ---- Merge league multipliers
    # Prefer competition_id exact merge if both sides have it; else fallback to normalized name
    try:
        mult_path = cfg.root_dir / cfg.league_multiplier_filename
        m = pd.read_excel(mult_path)
        # Normalise multiplier headers
        m_cols_lc = m.columns.str.strip().str.lower()
        m.columns = m_cols_lc

        # Accepted header variants
        # expected: 'competition_id' and/or 'league', 'multiplier'
        have_mult_col = "multiplier" in m.columns
        if not have_mult_col:
            raise ValueError("No 'multiplier' column in league multipliers file")

        merged = False

        if cfg.use_competition_id_when_present:
            if "competition_id" in df.columns and "competition_id" in m.columns:
                df = df.merge(m[["competition_id", "multiplier"]], on="competition_id", how="left")
                merged = True

        if not merged:
            # Normalise to lowercase for robust name join
            if COMPETITION_NORM in df.columns:
                df["_comp_norm_lc"] = _lc(df[COMPETITION_NORM])
            else:
                df["_comp_norm_lc"] = _lc(df.get(COMPETITION_COL, pd.Series(index=df.index, dtype="object")))

            league_key = None
            for cand in ("league", "competition", "competition_norm"):
                if cand in m.columns:
                    league_key = cand
                    break

            if league_key is not None:
                m["_league_lc"] = _lc(m[league_key])
                df = df.merge(m[["_league_lc", "multiplier"]], left_on="_comp_norm_lc", right_on="_league_lc", how="left")
                merged = True

        # Finalise Multiplier column
        if "multiplier" in df.columns:
            df[MULTIPLIER_COL] = pd.to_numeric(df["multiplier"], errors="coerce").fillna(1.0)
        elif MULTIPLIER_COL in df.columns:
            df[MULTIPLIER_COL] = pd.to_numeric(df[MULTIPLIER_COL], errors="coerce").fillna(1.0)
        else:
            df[MULTIPLIER_COL] = 1.0

        # Cleanup temp cols
        for c in ["_league_lc", "_comp_norm_lc", "multiplier"]:
            if c in df.columns:
                # keep a user-facing Multiplier only
                if c != MULTIPLIER_COL:
                    df.drop(columns=[c], inplace=True, errors="ignore")

        print(f"[DEBUG] ✅ League multipliers merged; unique weights: {df[MULTIPLIER_COL].nunique()}")
    except Exception as e:
        print(f"[DEBUG] ⚠️ League multipliers merge failed: {e}")
        if MULTIPLIER_COL not in df.columns:
            df[MULTIPLIER_COL] = 1.0
        df[MULTIPLIER_COL] = pd.to_numeric(df[MULTIPLIER_COL], errors="coerce").fillna(1.0)

    # ---- Build "Positions played"
    if cfg.build_positions_played:
        if "Position" in df.columns and "Secondary Position" in df.columns:
            p = df["Position"].fillna("").astype(str).str.strip()
            sp = df["Secondary Position"].fillna("").astype(str).str.strip()
            df["Positions played"] = p + np.where(sp != "", ", " + sp, "")
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
# Scoring
# ============================================================
def compute_scores(df_all_in: pd.DataFrame, cfg: ScoringConfig = ScoringConfig()) -> pd.DataFrame:
    """
    Team Rankings logic:
      - Eligible baseline: minutes >= cfg.min_minutes_for_baseline (fallback to full)
      - Z per metric per position (invert LOWER_IS_BETTER)
      - Weighted Z: +Z × multiplier, −Z ÷ multiplier
      - LFC variant: same formula but with special multiplier for target league
      - Per-position anchors (min/max of Weighted Z on eligible) → 0–100 scale
    """
    df_all = df_all_in.copy()

    # Ensure required columns
    for c, default in [(cfg.pos_col, np.nan), (cfg.minutes_col, np.nan), (cfg.multiplier_col, 1.0)]:
        if c not in df_all.columns:
            df_all[c] = default

    mins = pd.to_numeric(df_all.get(cfg.minutes_col, np.nan), errors="coerce").fillna(0)
    eligible = df_all[mins >= cfg.min_minutes_for_baseline].copy()
    if eligible.empty:
        eligible = df_all.copy()

    # Numeric metric candidates
    num_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude controls / non-performance fields
    exclude = {
        cfg.minutes_col, "Age", "Height",
        cfg.multiplier_col, LFC_MULTIPLIER_COL,
        cfg.avg_z_col, cfg.weighted_z_col, cfg.lfc_weighted_z_col,
        cfg.score_100_col, cfg.score_100_lfc_col,
    }
    # Also exclude common id/link columns that might be numeric
    exclude |= {"competition_id", "team_id", "player_id"}

    metric_cols = [c for c in num_cols if c not in exclude]
    if not metric_cols:
        # Nothing to score; create zero scores
        df_all[cfg.avg_z_col] = 0.0
        df_all[cfg.weighted_z_col] = 0.0
        df_all[cfg.score_100_col] = 50.0
        if cfg.lfc_enable:
            df_all[cfg.lfc_weighted_z_col] = 0.0
            df_all[cfg.score_100_lfc_col] = 50.0
        return df_all

    # Baseline mean/std from eligible only
    baseline_stats = (
        eligible
        .groupby(cfg.pos_col, dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .fillna(0)
    )
    baseline_stats.columns = baseline_stats.columns.map("_".join)  # "NP Goals_mean"

    raw_z = pd.DataFrame(index=df_all.index, columns=metric_cols, dtype=float)

    lower_better = set(cfg.lower_is_better)

    for m in metric_cols:
        mean_col = f"{m}_mean"
        std_col  = f"{m}_std"
        if mean_col not in baseline_stats.columns or std_col not in baseline_stats.columns:
            raw_z[m] = 0.0
            continue

        means = df_all[cfg.pos_col].map(baseline_stats[mean_col])
        stds  = df_all[cfg.pos_col].map(baseline_stats[std_col])
        stds  = stds.replace(0, 1)  # avoid /0

        z = (pd.to_numeric(df_all[m], errors="coerce") - means) / stds
        if m in lower_better:
            z *= -1.0
        raw_z[m] = z.fillna(0.0)

    df_all[cfg.avg_z_col] = raw_z.mean(axis=1).fillna(0.0)

    # Weighted Z (piecewise)
    mult = pd.to_numeric(df_all.get(cfg.multiplier_col, 1.0), errors="coerce").fillna(1.0)
    avgz = pd.to_numeric(df_all[cfg.avg_z_col], errors="coerce").fillna(0.0)

    df_all[cfg.weighted_z_col] = np.select(
        [avgz > 0, avgz < 0],
        [avgz * mult, avgz / mult],
        default=0.0,
    )

    # LFC variant
    if cfg.lfc_enable:
        lfc_mult = mult.copy()
        if COMPETITION_NORM in df_all.columns:
            lfc_mult = np.where(df_all[COMPETITION_NORM] == cfg.lfc_league_name,
                                float(cfg.lfc_multiplier_value), lfc_mult)
        df_all[LFC_MULTIPLIER_COL] = lfc_mult

        df_all[cfg.lfc_weighted_z_col] = np.select(
            [avgz > 0, avgz < 0],
            [avgz * lfc_mult, avgz / lfc_mult],
            default=0.0,
        )

    # Anchors from eligible, per position, using weighted Z
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

    # 0–100 scaling
    df_all[cfg.score_100_col] = [
        _to100(v, lo, hi) for v, lo, hi in zip(df_all[cfg.weighted_z_col],
                                               df_all["_scale_min"], df_all["_scale_max"])
    ]

    if cfg.lfc_enable and cfg.lfc_weighted_z_col in df_all.columns:
        df_all[cfg.score_100_lfc_col] = [
            _to100(v, lo, hi) for v, lo, hi in zip(df_all[cfg.lfc_weighted_z_col],
                                                   df_all["_scale_min"], df_all["_scale_max"])
        ]

    # Tidy
    df_all[cfg.score_100_col] = pd.to_numeric(df_all[cfg.score_100_col], errors="coerce").round(1).fillna(0)
    if cfg.lfc_enable and cfg.score_100_lfc_col in df_all.columns:
        df_all[cfg.score_100_lfc_col] = (
            pd.to_numeric(df_all[cfg.score_100_lfc_col], errors="coerce").round(1).fillna(0)
        )
    df_all.drop(columns=["_scale_min", "_scale_max"], inplace=True, errors="ignore")

    return df_all

# ============================================================
# Pipeline convenience
# ============================================================
def score_pipeline(
    df_in: pd.DataFrame,
    root_dir: Path,
    *,
    preprocess_config: Optional[PreprocessConfig] = None,
    scoring_config: Optional[ScoringConfig] = None,
) -> pd.DataFrame:
    if preprocess_config is None:
        preprocess_config = PreprocessConfig(root_dir=root_dir)
    if scoring_config is None:
        scoring_config = ScoringConfig()

    df_prep = preprocess_for_scoring(df_in, preprocess_config)
    df_scored = compute_scores(df_prep, scoring_config)
    return df_scored

__all__ = [
    "POS_COL", "MINUTES_COL", "COMPETITION_NORM", "MULTIPLIER_COL",
    "DEFAULT_LOWER_IS_BETTER",
    "PreprocessConfig", "ScoringConfig",
    "map_first_position_to_group",
    "preprocess_for_scoring",
    "compute_scores",
    "score_pipeline",
]
