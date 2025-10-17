# pages/3_Team_Rankings.py

import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime
import sqlite3

from auth import check_password
from branding import show_branding

# ===== NEW SHARED SCORING IMPORTS =====
from lib.scoring import (
    preprocess_for_scoring,
    compute_scores as scoring_compute_scores,
    PreprocessConfig,
    ScoringConfig,
)

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
# Main UI
# ============================================================
try:
    # ---------- Load and preprocess ----------
    df_all_raw = load_statsbomb(DATA_PATH)
    df_all_raw = add_age_column(df_all_raw)

    df_all = preprocess_for_scoring(
        df_all_raw,
        PreprocessConfig(root_dir=ROOT_DIR)
    )

    df_all = scoring_compute_scores(
        df_all,
        ScoringConfig(min_minutes_for_baseline=600)
    )

    # ---------- League and club selection ----------
    league_col = "Competition_norm"
    leagues = sorted(df_all[league_col].dropna().unique())
    if not leagues:
        st.error("No leagues found in data.")
        st.stop()

    selected_league = st.selectbox("Select League", leagues)
    clubs = sorted(df_all.loc[df_all[league_col] == selected_league, "Team"].dropna().unique())
    if not clubs:
        st.warning("No clubs found for this league.")
        st.stop()

    selected_club = st.selectbox("Select Club", clubs)

    df_team = df_all[(df_all[league_col] == selected_league) & (df_all["Team"] == selected_club)].copy()
    if df_team.empty:
        st.warning("No players found for this team.")
        st.stop()

    # ---------- Rank players within team ----------
    df_team["Rank in Team"] = df_team["Score (0–100)"].rank(ascending=False, method="min").astype(int)

    # ---------- Optional minutes filter ----------
    st.markdown("#### ⏱ Filter by Minutes Played (Display Only)")
    df_team["Minutes played"] = pd.to_numeric(df_team["Minutes played"], errors="coerce").fillna(0).astype(int)
    default_display_min = 600
    selected_min_display = st.number_input(
        "Show only players with at least this many minutes",
        min_value=0,
        max_value=int(df_team["Minutes played"].max()),
        value=default_display_min,
        step=50,
        key="display_minutes_input"
    )
    df_team = df_team[df_team["Minutes played"] >= selected_min_display].copy()
    if df_team.empty:
        st.warning(f"No players with ≥ {selected_min_display} minutes in this team.")
        st.stop()

    # ---------- Team average scores ----------
    avg_score = df_team["Score (0–100)"].mean() if "Score (0–100)" in df_team.columns else np.nan
    avg_lfc = df_team["LFC Score (0–100)"].mean() if "LFC Score (0–100)" in df_team.columns else np.nan

    if not np.isnan(avg_score):
        if not np.isnan(avg_lfc):
            st.markdown(f"### {selected_club} ({selected_league}) — Average {avg_score:.1f} ({avg_lfc:.1f} LFC Score)")
        else:
            st.markdown(f"### {selected_club} ({selected_league}) — Average {avg_score:.1f}")
    else:
        st.markdown(f"### {selected_club} ({selected_league}) — No eligible players")

    # ---------- Table ----------
    cols_for_table = [
        "Player", "Six-Group Position", "Positions played",
        "Team", league_col, "Multiplier",
        "Avg Z Score", "Weighted Z Score", "LFC Weighted Z",
        "Score (0–100)", "LFC Score (0–100)",
        "Age", "Minutes played", "Rank in Team"
    ]
    for c in cols_for_table:
        if c not in df_team.columns:
            df_team[c] = np.nan

    z_ranking = df_team[cols_for_table].copy()
    z_ranking.rename(columns={
        "Six-Group Position": "Position",
        league_col: "League",
        "Multiplier": "League Weight",
        "Avg Z Score": "Z Avg",
        "Weighted Z Score": "Z Weighted",
        "LFC Weighted Z": "Z LFC Weighted"
    }, inplace=True)

    z_ranking["Age"] = pd.to_numeric(z_ranking["Age"], errors="coerce").round(0)
    z_ranking["Minutes played"] = pd.to_numeric(z_ranking["Minutes played"], errors="coerce").fillna(0).astype(int)
    z_ranking["League Weight"] = pd.to_numeric(z_ranking["League Weight"], errors="coerce").fillna(1.0).round(3)

    # ---------- Favourites integration ----------
    favs_in_db = {row[0] for row in get_favourites()}
    z_ranking["⭐ Favourite"] = z_ranking["Player"].isin(favs_in_db)

    edited = st.data_editor(
        z_ranking,
        column_config={
            "⭐ Favourite": st.column_config.CheckboxColumn("⭐ Favourite", help="Mark as favourite"),
            "League Weight": st.column_config.NumberColumn("League Weight", help="League weighting applied in ranking", format="%.3f"),
            "Z Avg": st.column_config.NumberColumn("Z Avg", format="%.3f"),
            "Z Weighted": st.column_config.NumberColumn("Z Weighted", format="%.3f"),
            "Z LFC Weighted": st.column_config.NumberColumn("Z LFC Weighted", format="%.3f"),
            "Score (0–100)": st.column_config.NumberColumn("Score (0–100)", format="%.1f"),
            "LFC Score (0–100)": st.column_config.NumberColumn("LFC Score (0–100)", format="%.1f"),
        },
        hide_index=False,
        width="stretch",
    )

    # ---------- Update favourites ----------
    for _, r in edited.iterrows():
        p = r["Player"]
        if r["⭐ Favourite"] and p not in favs_in_db:
            add_favourite(p, r.get("Team"), r.get("League"), r.get("Positions played"))
        elif not r["⭐ Favourite"] and p in favs_in_db:
            remove_favourite(p)

except Exception as e:
    st.error(f"❌ Could not load data: {e}")
