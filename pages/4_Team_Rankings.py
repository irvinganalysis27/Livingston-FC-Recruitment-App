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
    conn.execute(
        "INSERT OR REPLACE INTO favourites (player, team, league, position) VALUES (?,?,?,?)",
        (player, team, league, position)
    )
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
# Position Mapping
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
FRIENDLY_TO_GROUP = {
    "FULLBACK": "Full Back", "FULL BACK": "Full Back",
    "CENTREBACK": "Centre Back", "CENTRE BACK": "Centre Back",
    "NUMBER6": "Number 6", "NO6": "Number 6", "DM": "Number 6",
    "NUMBER8": "Number 8", "NO8": "Number 8", "CM": "Number 8",
    "WINGER": "Winger", "STRIKER": "Striker", "FORWARD": "Striker",
    "GOALKEEPER": "Goalkeeper", "GK": "Goalkeeper",
}

def _clean_pos_token(tok: str) -> str:
    if pd.isna(tok): return ""
    t = str(tok).upper().strip()
    t = re.sub(r"[.\-_/]", " ", t)
    return re.sub(r"\s+", " ", t)

def map_first_position_to_group(primary_pos_cell) -> str:
    t = _clean_pos_token(primary_pos_cell)
    sb_key = t.replace(" ", "")
    if sb_key in RAW_TO_GROUP:
        return RAW_TO_GROUP[sb_key]
    if t in FRIENDLY_TO_GROUP:
        return FRIENDLY_TO_GROUP[t]
    if sb_key in FRIENDLY_TO_GROUP:
        return FRIENDLY_TO_GROUP[sb_key]
    return None

# ============================================================
# Position Metric Templates
# ============================================================
position_metrics = {
    "Goalkeeper": [
        "Goals Conceded", "PSxG Faced", "GSAA", "Save%", "xSv%", "Shot Stopping%",
        "Shots Faced", "Shots Faced OT%", "Positive Outcome%", "Goalkeeper OBV"
    ],
    "Centre Back": [
        "Passing%", "Pass OBV", "Pr. Long Balls", "UPr. Long Balls", "OBV",
        "PAdj Interceptions", "PAdj Tackles", "Dribbles Stopped%",
        "Defensive Actions", "Aggressive Actions", "Fouls", "Aerial Wins", "Aerial Win%"
    ],
    "Full Back": [
        "Passing%", "Pr. Pass% Dif.", "Successful Box Cross%", "Deep Progressions",
        "Successful Dribbles", "Turnovers", "OBV", "Pass OBV",
        "Defensive Actions", "Aerial Win%", "PAdj Pressures", "PAdj Tack&Int"
    ],
    "Number 6": [
        "xGBuildup", "xG Assisted", "Passing%", "Deep Progressions",
        "Turnovers", "OBV", "Pass OBV", "Pr. Pass% Dif.",
        "PAdj Interceptions", "PAdj Tackles", "Dribbles Stopped%",
        "Aggressive Actions", "Aerial Win%", "Pressure Regains"
    ],
    "Number 8": [
        "xGBuildup", "xG Assisted", "Shots", "xG", "NP Goals",
        "Passing%", "Deep Progressions", "OP Passes Into Box", "Pass OBV", "OBV",
        "Pressure Regains", "PAdj Pressures", "Aggressive Actions"
    ],
    "Winger": [
        "xG", "Shots", "xG/Shot", "Touches In Box", "OP xG Assisted",
        "NP Goals", "OP Passes Into Box", "Successful Box Cross%",
        "Passing%", "Successful Dribbles", "Turnovers", "OBV", "D&C OBV", "Fouls Won"
    ],
    "Striker": [
        "NP Goals", "xG", "Shots", "xG/Shot", "Goal Conversion%",
        "Touches In Box", "xG Assisted", "Fouls Won", "Deep Completions",
        "OP Key Passes", "Aerial Win%", "Aerial Wins"
    ]
}

LOWER_IS_BETTER = {"Turnovers", "Fouls", "Pr. Long Balls", "UPr. Long Balls"}

# ============================================================
# Preprocessing
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
# Compute Scores (League + Position)
# ============================================================
def compute_scores(df_all: pd.DataFrame, min_minutes: int = 600) -> pd.DataFrame:
    df = df_all.copy()
    pos_col = "Six-Group Position"
    league_col = "Competition_norm"

    df["Minutes played"] = pd.to_numeric(df.get("Minutes played", 0), errors="coerce").fillna(0)
    eligible = df[df["Minutes played"] >= min_minutes].copy()
    if eligible.empty:
        eligible = df.copy()

    df["Avg Z Score"] = 0.0
    df["Weighted Z Score"] = 0.0
    df["Score (0–100)"] = 50.0

    for (league, position), idx in df.groupby([league_col, pos_col]).groups.items():
        if pd.isna(league) or pd.isna(position):
            continue

        metrics = position_metrics.get(position, [])
        existing = [m for m in metrics if m in df.columns]
        if not existing:
            continue

        for m in existing:
            df[m] = pd.to_numeric(df[m], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

        elig_mask = (eligible[league_col] == league) & (eligible[pos_col] == position)
        elig_pos = eligible.loc[elig_mask, existing]
        if elig_pos.empty:
            elig_pos = eligible.loc[eligible[league_col] == league, existing]
        if elig_pos.empty:
            elig_pos = eligible[existing]

        mean_vals = elig_pos.mean()
        std_vals = elig_pos.std().replace(0, 1)

        mask_lp = (df[league_col] == league) & (df[pos_col] == position)
        Z = ((df.loc[mask_lp, existing] - mean_vals) / std_vals).fillna(0)
        for m in (LOWER_IS_BETTER & set(existing)):
            Z[m] = -Z[m]
        df.loc[mask_lp, "Avg Z Score"] = Z.mean(axis=1).astype(float)

    mult = pd.to_numeric(df.get("Multiplier", 1.0), errors="coerce").fillna(1.0)
    az = df["Avg Z Score"]
    df["Weighted Z Score"] = np.where(az >= 0, az * mult, az / mult)

    anchors = (
        df.groupby([league_col, pos_col], dropna=False)["Weighted Z Score"]
        .agg(_lo="min", _hi="max").reset_index()
    )
    df = df.merge(anchors, on=[league_col, pos_col], how="left")

    def to100(v, lo, hi):
        if pd.isna(v) or pd.isna(lo) or pd.isna(hi) or hi <= lo:
            return 50.0
        return float(np.clip((v - lo) / (hi - lo) * 100.0, 0.0, 100.0))

    df["Score (0–100)"] = [
        to100(v, lo, hi) for v, lo, hi in zip(df["Weighted Z Score"], df["_lo"], df["_hi"])
    ]
    df.drop(columns=["_lo", "_hi"], inplace=True, errors="ignore")
    return df

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

    df_team = df_all[
        (df_all[league_col] == selected_league)
        & (df_all["Team"] == selected_club)
    ].copy()
    if df_team.empty:
        st.warning("No players found for this team.")
        st.stop()

    df_team["Minutes played"] = (
        pd.to_numeric(df_team["Minutes played"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)
    )
    df_team["Score (0–100)"] = pd.to_numeric(df_team["Score (0–100)"], errors="coerce").fillna(0)

    df_team["Rank in Team"] = (
        df_team["Score (0–100)"].rank(ascending=False, method="min").fillna(0).astype(int)
    )

    st.markdown("#### ⏱ Filter by Minutes Played (Display Only)")
    min_val, max_val = int(df_team["Minutes played"].min()), int(df_team["Minutes played"].max())
    default_display_min = min(600, max_val)
    selected_min_display = st.number_input(
        "Show only players with at least this many minutes",
        min_value=min_val, max_value=max_val, value=default_display_min, step=50
    )
    df_team = df_team[df_team["Minutes played"] >= selected_min_display].copy()

    avg_score = df_team["Score (0–100)"].mean() if not df_team.empty else np.nan
    st.markdown(f"### {selected_club} ({selected_league}) — Average {avg_score:.1f}" if not np.isnan(avg_score)
                else f"### {selected_club} ({selected_league}) — No eligible players")

    cols_for_table = [
        "Player", "Six-Group Position", "Positions played", "Team", league_col, "Multiplier",
        "Avg Z Score", "Weighted Z Score", "Score (0–100)", "Age", "Minutes played", "Rank in Team"
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
        "Weighted Z Score": "Z Weighted"
    }, inplace=True)

    favs_in_db = {row[0] for row in get_favourites()}
    z_ranking["⭐ Favourite"] = z_ranking["Player"].isin(favs_in_db)

    edited = st.data_editor(
        z_ranking,
        column_config={
            "⭐ Favourite": st.column_config.CheckboxColumn("⭐ Favourite", help="Mark as favourite"),
            "League Weight": st.column_config.NumberColumn("League Weight", format="%.3f"),
            "Z Avg": st.column_config.NumberColumn("Z Avg", format="%.3f"),
            "Z Weighted": st.column_config.NumberColumn("Z Weighted", format="%.3f"),
            "Score (0–100)": st.column_config.NumberColumn("Score (0–100)", format="%.1f"),
        },
        hide_index=False,
        width="stretch",
    )

    for _, r in edited.iterrows():
        p = r["Player"]
        if r["⭐ Favourite"] and p not in favs_in_db:
            add_favourite(p, r.get("Team"), r.get("League"), r.get("Position"))
        elif not r["⭐ Favourite"] and p in favs_in_db:
            remove_favourite(p)

except Exception as e:
    st.error(f"❌ Could not load data: {e}")
