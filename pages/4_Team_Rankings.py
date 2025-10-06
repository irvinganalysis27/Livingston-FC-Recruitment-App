# pages/4_Team_Rankings.py

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from auth import check_password
from branding import show_branding

# ============================================================
# Protect page
# ============================================================
if not check_password():
    st.stop()

show_branding()
st.title("🏆 Team Player Rankings")

APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent
DATA_PATH = ROOT_DIR / "statsbomb_player_stats_clean.csv"

# ============================================================
# Utility functions
# ============================================================

def load_one_file(p: Path) -> pd.DataFrame:
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    else:
        return pd.read_csv(p)

def load_statsbomb(path: Path) -> pd.DataFrame:
    if path.is_file():
        return load_one_file(path)
    else:
        files = sorted([f for f in path.iterdir() if f.is_file()])
        frames = [load_one_file(f) for f in files]
        return pd.concat(frames, ignore_index=True, sort=False)

def add_age_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Birth Date" in df.columns:
        today = datetime.today()
        df["Age"] = pd.to_datetime(df["Birth Date"], errors="coerce").apply(
            lambda dob: today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            if pd.notna(dob) else np.nan
        )
    return df

# ============================================================
# Position & metric mapping
# ============================================================

RAW_TO_GROUP = {
    "LEFTBACK": "Full Back", "LEFTWINGBACK": "Full Back",
    "RIGHTBACK": "Full Back", "RIGHTWINGBACK": "Full Back",
    "CENTREBACK": "Centre Back", "LEFTCENTREBACK": "Centre Back", "RIGHTCENTREBACK": "Centre Back",
    "DEFENSIVEMIDFIELDER": "Number 6", "LEFTDEFENSIVEMIDFIELDER": "Number 6",
    "RIGHTDEFENSIVEMIDFIELDER": "Number 6", "CENTREDEFENSIVEMIDFIELDER": "Number 6",
    "CENTREMIDFIELDER": "Number 8", "LEFTCENTREMIDFIELDER": "Number 8", "RIGHTCENTREMIDFIELDER": "Number 8",
    "CENTREATTACKINGMIDFIELDER": "Number 8", "SECONDSTRIKER": "Number 8",
    "LEFTWING": "Winger", "LEFTMIDFIELDER": "Winger",
    "RIGHTWING": "Winger", "RIGHTMIDFIELDER": "Winger",
    "CENTREFORWARD": "Striker", "LEFTCENTREFORWARD": "Striker", "RIGHTCENTREFORWARD": "Striker",
    "GOALKEEPER": "Goalkeeper",
}

def _clean_pos_token(tok: str) -> str:
    if pd.isna(tok):
        return ""
    t = str(tok).upper().strip()
    t = re.sub(r"[.\-_/]", " ", t)
    t = re.sub(r"\s+", "", t)
    return t

def map_first_position_to_group(primary_pos_cell) -> str:
    tok = _clean_pos_token(primary_pos_cell)
    return RAW_TO_GROUP.get(tok, None)

LOWER_IS_BETTER = {"Turnovers","Fouls","Pr. Long Balls","UPr. Long Balls"}

# ============================================================
# Compute player rankings
# ============================================================

def compute_rankings(df_all: pd.DataFrame, min_minutes: int = 600) -> pd.DataFrame:
    pos_col = "Six-Group Position"
    if pos_col not in df_all.columns:
        df_all[pos_col] = np.nan

    # Collect numeric metrics (simplified detection)
    numeric_cols = [c for c in df_all.columns if pd.api.types.is_numeric_dtype(df_all[c])]
    raw_z_all = pd.DataFrame(index=df_all.index, columns=numeric_cols, dtype=float)

    for m in numeric_cols:
        z_per_group = df_all.groupby(pos_col)[m].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
        )
        if m in LOWER_IS_BETTER:
            z_per_group *= -1
        raw_z_all[m] = z_per_group.fillna(0)

    df_all["Avg Z Score"] = raw_z_all.mean(axis=1).fillna(0)

    # Apply league multiplier if exists
    if "Multiplier" not in df_all.columns:
        df_all["Multiplier"] = 1.0
    df_all["Multiplier"] = pd.to_numeric(df_all["Multiplier"], errors="coerce").fillna(1.0)
    df_all["Weighted Z Score"] = df_all["Avg Z Score"] * df_all["Multiplier"]

    eligible = df_all[pd.to_numeric(df_all.get("Minutes played", np.nan), errors="coerce").fillna(0) >= min_minutes].copy()
    if eligible.empty:
        eligible = df_all.copy()

    anchor_minmax = (
        eligible.groupby(pos_col)["Weighted Z Score"].agg(_scale_min="min", _scale_max="max").fillna(0)
    )
    df_all = df_all.merge(anchor_minmax, left_on=pos_col, right_index=True, how="left")

    def _minmax_score(val, lo, hi):
        if pd.isna(val) or pd.isna(lo) or pd.isna(hi) or hi <= lo:
            return 50.0
        return np.clip((val - lo) / (hi - lo) * 100.0, 0.0, 100.0)

    df_all["Score (0–100)"] = [
        _minmax_score(v, lo, hi)
        for v, lo, hi in zip(df_all["Weighted Z Score"], df_all["_scale_min"], df_all["_scale_max"])
    ]
    df_all["Score (0–100)"] = pd.to_numeric(df_all["Score (0–100)"], errors="coerce").round(1).fillna(0)
    df_all.drop(columns=["_scale_min", "_scale_max"], inplace=True, errors="ignore")

    df_all["Rank"] = df_all["Score (0–100)"].rank(ascending=False, method="min").astype(int)
    return df_all

# ============================================================
# Main UI
# ============================================================

try:
    df_all_raw = load_statsbomb(DATA_PATH)
    df_all_raw = add_age_column(df_all_raw)

    df_all = df_all_raw.copy()
    if "Name" in df_all.columns:
        df_all.rename(columns={"Name": "Player"}, inplace=True)
    if "Primary Position" in df_all.columns:
        df_all.rename(columns={"Primary Position": "Position"}, inplace=True)
    if "Minutes" in df_all.columns:
        df_all.rename(columns={"Minutes": "Minutes played"}, inplace=True)

    df_all["Six-Group Position"] = df_all["Position"].apply(map_first_position_to_group)
    df_all = compute_rankings(df_all)

    # --- League filter ---
    league_col = "Competition_norm" if "Competition_norm" in df_all.columns else "Competition"
    leagues = sorted(df_all[league_col].dropna().unique())

    selected_league = st.selectbox("Select League", leagues)

    # --- Club filter ---
    clubs = sorted(df_all.loc[df_all[league_col] == selected_league, "Team"].dropna().unique())
    selected_club = st.selectbox("Select Club", clubs)

    # --- Filter for team ---
    df_team = df_all[(df_all[league_col] == selected_league) & (df_all["Team"] == selected_club)].copy()

    if df_team.empty:
        st.warning("No players found for this team.")
        st.stop()

    # --- Sort by Score ---
    df_team = df_team.sort_values("Score (0–100)", ascending=False).reset_index(drop=True)
    df_team["Rank in Team"] = df_team["Score (0–100)"].rank(ascending=False, method="min").astype(int)

    # --- Display table ---
    display_cols = ["Rank in Team", "Player", "Six-Group Position", "Score (0–100)",
                    "Age", "Minutes played", "Weighted Z Score"]
    for c in display_cols:
        if c not in df_team.columns:
            df_team[c] = np.nan

    st.markdown(f"### {selected_club} ({selected_league}) — Player Rankings")
    st.data_editor(
        df_team[display_cols],
        hide_index=True,
        width="stretch",
        column_config={
            "Score (0–100)": st.column_config.NumberColumn("Score (0–100)", format="%.1f"),
            "Weighted Z Score": st.column_config.NumberColumn("Weighted Z", format="%.2f"),
        },
    )

except Exception as e:
    st.error(f"❌ Could not load data: {e}")
