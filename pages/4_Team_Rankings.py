# pages/4_Team_Rankings.py

import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime
import sqlite3

from auth import check_password
from branding import show_branding

# ============================================================
# Setup & Protection
# ============================================================
if not check_password():
    st.stop()

show_branding()
st.title("🏆 Team Player Rankings")

APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent
DATA_PATH = ROOT_DIR / "statsbomb_player_stats_clean.csv"
DB_PATH = APP_DIR / "favourites.db"

# ============================================================
# Favourites DB
# ============================================================
def _ensure_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS favourites (
            player TEXT PRIMARY KEY,
            team TEXT,
            league TEXT,
            position TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
_ensure_db()

def get_favourites():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT player, team, league, position FROM favourites").fetchall()
    conn.close()
    return rows

def add_favourite(player, team=None, league=None, position=None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT OR REPLACE INTO favourites (player, team, league, position) VALUES (?,?,?,?)",
                 (player, team, league, position))
    conn.commit()
    conn.close()

def remove_favourite(player):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM favourites WHERE player=?", (player,))
    conn.commit()
    conn.close()

# ============================================================
# Data Loading
# ============================================================
def load_one_file(p: Path) -> pd.DataFrame:
    return pd.read_excel(p) if p.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(p)

def load_statsbomb(path: Path) -> pd.DataFrame:
    if path.is_file():
        return load_one_file(path)
    frames = [load_one_file(f) for f in sorted(p for p in path.iterdir() if p.is_file())]
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
# Position & League Mapping
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
def _clean_pos_token(tok: str) -> str:
    if pd.isna(tok): return ""
    t = str(tok).upper().strip()
    t = re.sub(r"[.\-_/]", " ", t)
    return re.sub(r"\s+", "", t)

def map_first_position_to_group(primary_pos_cell) -> str:
    return RAW_TO_GROUP.get(_clean_pos_token(primary_pos_cell), None)

LOWER_IS_BETTER = {"Turnovers", "Fouls", "Pr. Long Balls", "UPr. Long Balls"}

# ============================================================
# Preprocessing (League + Multiplier + Position)
# ============================================================
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {}
    if "Name" in df.columns: rename_map["Name"] = "Player"
    if "Primary Position" in df.columns: rename_map["Primary Position"] = "Position"
    if "Minutes" in df.columns: rename_map["Minutes"] = "Minutes played"
    if rename_map: df.rename(columns=rename_map, inplace=True)

    if "Competition_norm" not in df.columns and "Competition" in df.columns:
        df["Competition_norm"] = df["Competition"].astype(str)

    # merge multipliers
    try:
        mult = pd.read_excel(ROOT_DIR / "league_multipliers.xlsx")
        if {"League", "Multiplier"}.issubset(mult.columns):
            df = df.merge(mult, left_on="Competition_norm", right_on="League", how="left")
            df["Multiplier"] = pd.to_numeric(df["Multiplier"], errors="coerce").fillna(1.0)
        else:
            df["Multiplier"] = 1.0
    except Exception:
        df["Multiplier"] = 1.0

    if "Secondary Position" in df.columns:
        df["Positions played"] = df["Position"].fillna("") + np.where(
            df["Secondary Position"].notna(), ", " + df["Secondary Position"].astype(str), ""
        )
    else:
        df["Positions played"] = df["Position"]

    df["Six-Group Position"] = df["Position"].apply(map_first_position_to_group)
    return df

# ============================================================
# ✅ Position-specific scoring logic (IDENTICAL TO RADAR PAGE)
# ============================================================

from pages.Player_Radar import position_metrics, LOWER_IS_BETTER  # adjust filename if needed

def compute_scores(df_all: pd.DataFrame, min_minutes: int = 600) -> pd.DataFrame:
    """
    Compute position-specific 0–100 scores identical to the Radar Page logic.
    - Uses metrics defined in position_metrics.
    - Z-scores calculated within Six-Group Position.
    - League multiplier applied before scaling.
    - Anchors (min/max) come from players ≥ min_minutes.
    """
    df_all = df_all.copy()
    pos_col = "Six-Group Position"

    # Ensure numeric conversion for relevant columns
    for col in df_all.columns:
        if not pd.api.types.is_numeric_dtype(df_all[col]):
            try:
                df_all[col] = pd.to_numeric(df_all[col], errors="coerce")
            except Exception:
                pass

    df_all["Avg Z Score"] = np.nan
    df_all["Weighted Z Score"] = np.nan

    results = []

    # Loop through each position group using the radar’s metric sets
    for pos, meta in position_metrics.items():
        metrics = [m for m in meta["metrics"] if m in df_all.columns]
        subset = df_all[df_all[pos_col] == pos].copy()
        if subset.empty or len(metrics) == 0:
            continue

        raw_z = pd.DataFrame(index=subset.index, columns=metrics, dtype=float)
        for m in metrics:
            subset[m] = pd.to_numeric(subset[m], errors="coerce").fillna(0)
            z = subset[m].transform(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0)
            if m in LOWER_IS_BETTER:
                z *= -1
            raw_z[m] = z.fillna(0)

        subset["Avg Z Score"] = raw_z.mean(axis=1).fillna(0)
        subset["Multiplier"] = pd.to_numeric(subset.get("Multiplier", 1.0), errors="coerce").fillna(1.0)
        subset["Weighted Z Score"] = subset["Avg Z Score"] * subset["Multiplier"]

        results.append(subset)

    if results:
        df_all = pd.concat(results, ignore_index=True)
    else:
        st.warning("No valid positional data to compute scores.")
        return df_all

    # --- Anchor scaling (same as radar) ---
    mins = pd.to_numeric(df_all.get("Minutes played", np.nan), errors="coerce").fillna(0)
    eligible = df_all[mins >= min_minutes].copy()
    if eligible.empty:
        eligible = df_all.copy()

    anchors = (
        eligible.groupby(pos_col)["Weighted Z Score"]
                .agg(_scale_min="min", _scale_max="max")
                .fillna(0)
    )

    df_all = df_all.merge(anchors, left_on=pos_col, right_index=True, how="left")

    # --- Scale Weighted Z Score → 0–100 (identical to radar) ---
    def _to100(v, lo, hi):
        try:
            v, lo, hi = float(v), float(lo), float(hi)
            if np.isnan(v) or np.isnan(lo) or np.isnan(hi) or hi <= lo:
                return 50.0
            return np.clip((v - lo) / (hi - lo) * 100.0, 0.0, 100.0)
        except Exception:
            return 50.0

    df_all["Score (0–100)"] = [
        _to100(v, lo, hi)
        for v, lo, hi in zip(df_all["Weighted Z Score"], df_all["_scale_min"], df_all["_scale_max"])
    ]
    df_all["Score (0–100)"] = pd.to_numeric(df_all["Score (0–100)"], errors="coerce").round(1).fillna(0)

    df_all.drop(columns=["_scale_min", "_scale_max"], inplace=True, errors="ignore")

    return df_all

# ============================================================
# Main UI
# ============================================================
try:
    df_all_raw = load_statsbomb(DATA_PATH)
    df_all_raw = add_age_column(df_all_raw)
    df_all = preprocess(df_all_raw)
    df_all = compute_scores(df_all, min_minutes=600)

    league_col = "Competition_norm"
    leagues = sorted(df_all[league_col].dropna().unique())
    selected_league = st.selectbox("Select League", leagues)

    clubs = sorted(df_all.loc[df_all[league_col] == selected_league, "Team"].dropna().unique())
    selected_club = st.selectbox("Select Club", clubs)

    df_team = df_all[(df_all[league_col] == selected_league) & (df_all["Team"] == selected_club)].copy()
    if df_team.empty:
        st.warning("No players found for this team.")
        st.stop()

    # ---------- Minutes filter ----------
    st.markdown("#### ⏱ Filter by Minutes Played")

    # Ensure the column exists and is numeric
    df_team["Minutes played"] = pd.to_numeric(df_team["Minutes played"], errors="coerce").fillna(0).astype(int)

    # Persist the minutes filter across reruns
    if "team_min_minutes" not in st.session_state:
        st.session_state.team_min_minutes = 600  # default like radar page

    st.session_state.team_min_minutes = st.number_input(
        "Minimum minutes to include",
        min_value=0,
        max_value=int(df_team["Minutes played"].max()),
        value=st.session_state.team_min_minutes,
        step=50,
        key="team_min_minutes_input"
    )

    # Apply the filter
    min_minutes = st.session_state.team_min_minutes
    df_team = df_team[df_team["Minutes played"] >= min_minutes].copy()

    if df_team.empty:
        st.warning(f"No players with ≥ {min_minutes} minutes.")
        st.stop()
        
    df_team["Rank in Team"] = df_team["Score (0–100)"].rank(ascending=False, method="min").astype(int)

    # ---------- Table ----------
    st.markdown(f"### {selected_club} ({selected_league}) — Players Ranked by Score (0–100)")

    cols_for_table = [
        "Player", "Six-Group Position", "Positions played",
        "Team", league_col, "Multiplier",
        "Score (0–100)", "Age", "Minutes played", "Rank in Team"
    ]
    for c in cols_for_table:
        if c not in df_team.columns: df_team[c] = np.nan

    z_ranking = df_team[cols_for_table].copy()
    z_ranking.rename(columns={
        "Six-Group Position": "Position",
        league_col: "League",
        "Multiplier": "League Weight"
    }, inplace=True)

    z_ranking["Age"] = pd.to_numeric(z_ranking["Age"], errors="coerce").round(0)
    z_ranking["Minutes played"] = pd.to_numeric(z_ranking["Minutes played"], errors="coerce").fillna(0).astype(int)
    z_ranking["League Weight"] = pd.to_numeric(z_ranking["League Weight"], errors="coerce").fillna(1.0).round(3)

    favs_in_db = {row[0] for row in get_favourites()}
    z_ranking["⭐ Favourite"] = z_ranking["Player"].isin(favs_in_db)

    edited = st.data_editor(
        z_ranking,
        column_config={
            "⭐ Favourite": st.column_config.CheckboxColumn("⭐ Favourite", help="Mark as favourite"),
            "League Weight": st.column_config.NumberColumn("League Weight", help="League weighting applied in ranking", format="%.3f"),
            "Score (0–100)": st.column_config.NumberColumn("Score (0–100)", format="%.1f"),
        },
        hide_index=False,
        width="stretch",
    )

    for _, r in edited.iterrows():
        p = r["Player"]
        if r["⭐ Favourite"] and p not in favs_in_db:
            add_favourite(p, r.get("Team"), r.get("League"), r.get("Positions played"))
        elif not r["⭐ Favourite"] and p in favs_in_db:
            remove_favourite(p)

except Exception as e:
    st.error(f"❌ Could not load data: {e}")
