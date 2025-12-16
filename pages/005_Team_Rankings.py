import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime
import sqlite3

from auth import check_password
from branding import show_branding

st.set_page_config(page_title="Livingston FC Recruitment App", layout="centered")

st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Authentication ----------
if not check_password():
    st.stop()

# ---------- Branding ----------
show_branding()
st.title("Team Player Rankings")

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
    frames = [load_one_file(f) for f in sorted(path.iterdir()) if f.is_file()]
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
    "LEFTBACK": "Full Back", "RIGHTBACK": "Full Back", "LEFTWINGBACK": "Full Back", "RIGHTWINGBACK": "Full Back",
    "CENTREBACK": "Centre Back", "LEFTCENTREBACK": "Centre Back", "RIGHTCENTREBACK": "Centre Back",
    "DEFENSIVEMIDFIELDER": "Number 6", "LEFTDEFENSIVEMIDFIELDER": "Number 6",
    "RIGHTDEFENSIVEMIDFIELDER": "Number 6", "CENTREDEFENSIVEMIDFIELDER": "Number 6",
    "CENTREMIDFIELDER": "Centre Midfield", "LEFTCENTREMIDFIELDER": "Centre Midfield", "RIGHTCENTREMIDFIELDER": "Centre Midfield",
    "CENTREATTACKINGMIDFIELDER": "Number 8", "RIGHTATTACKINGMIDFIELDER": "Number 8", "LEFTATTACKINGMIDFIELDER": "Number 8",
    "SECONDSTRIKER": "Number 8",
    "LEFTWING": "Winger", "RIGHTWING": "Winger",
    "LEFTMIDFIELDER": "Winger", "RIGHTMIDFIELDER": "Winger",
    "CENTREFORWARD": "Striker", "LEFTCENTREFORWARD": "Striker", "RIGHTCENTREFORWARD": "Striker",
    "GOALKEEPER": "Goalkeeper"
}

def _clean_pos_token(tok: str) -> str:
    if pd.isna(tok): return ""
    t = str(tok).upper().strip()
    t = re.sub(r"[.\-_/]", " ", t)
    return re.sub(r"\s+", "", t)

def map_first_position_to_group(primary_pos_cell) -> str:
    return RAW_TO_GROUP.get(_clean_pos_token(primary_pos_cell), None)

# ============================================================
# Metric Templates
# ============================================================
position_metrics = {
    "Goalkeeper": {
        "metrics": [
            "Goals Conceded", "PSxG Faced", "GSAA", "Save%", "xSv%", "Shot Stopping%",
            "Shots Faced", "Shots Faced OT%", "Positive Outcome%", "Goalkeeper OBV"
        ]
    },
    "Centre Back": {
        "metrics": [
            "Passing%", "Pass OBV", "Pr. Long Balls", "UPr. Long Balls", "OBV", "Pr. Pass% Dif.",
            "PAdj Interceptions", "PAdj Tackles", "Dribbles Stopped%", "Defensive Actions",
            "Aggressive Actions", "Fouls", "Aerial Wins", "Aerial Win%"
        ]
    },
    "Full Back": {
        "metrics": [
            "Passing%", "Pr. Pass% Dif.", "Successful Box Cross%", "Deep Progressions",
            "Successful Dribbles", "Turnovers", "OBV", "Pass OBV", "Defensive Actions",
            "Aerial Win%", "PAdj Pressures", "PAdj Tack&Int", "Dribbles Stopped%", "Aggressive Actions"
        ]
    },
    "Number 6": {
        "metrics": [
            "xGBuildup", "xG Assisted", "Passing%", "Deep Progressions", "Turnovers", "OBV", "Pass OBV",
            "Pr. Pass% Dif.", "PAdj Interceptions", "PAdj Tackles", "Dribbles Stopped%",
            "Aggressive Actions", "Aerial Win%", "Pressure Regains"
        ]
    },
    "Number 8": {
        "metrics": [
            "xGBuildup", "xG Assisted", "Shots", "xG", "NP Goals", "Passing%", "Deep Progressions",
            "OP Passes Into Box", "Pass OBV", "OBV", "Pressure Regains", "PAdj Pressures", "Aggressive Actions"
        ]
    },
    "Winger": {
        "metrics": [
            "xG", "Shots", "xG/Shot", "Touches In Box", "OP xG Assisted", "NP Goals",
            "OP Passes Into Box", "Successful Box Cross%", "Passing%", "Successful Dribbles",
            "Turnovers", "OBV", "D&C OBV", "Fouls Won", "Deep Progressions"
        ]
    },
    "Striker": {
        "metrics": [
            "NP Goals", "xG", "Shots", "xG/Shot", "Goal Conversion%", "Touches In Box",
            "xG Assisted", "Fouls Won", "Deep Completions", "OP Key Passes", "Aerial Win%",
            "Aerial Wins", "Aggressive Actions"
        ]
    }
}

LOWER_IS_BETTER = {"Turnovers", "Fouls", "Pr. Long Balls", "UPr. Long Balls"}

# ============================================================
# Preprocessing
# ============================================================
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {"Name": "Player", "Primary Position": "Position", "Minutes": "Minutes played"}
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    if "Competition_norm" not in df.columns and "Competition" in df.columns:
        df["Competition_norm"] = df["Competition"]

    # Merge league multipliers
    try:
        mult = pd.read_excel(ROOT_DIR / "league_multipliers.xlsx")
        if {"League", "Multiplier"}.issubset(mult.columns):
            df = df.merge(mult, left_on="Competition_norm", right_on="League", how="left")
            df["Multiplier"] = pd.to_numeric(df["Multiplier"], errors="coerce").fillna(1.0)
        else:
            df["Multiplier"] = 1.0
    except Exception:
        df["Multiplier"] = 1.0

    # Position mapping
    df["Six-Group Position"] = df["Position"].apply(map_first_position_to_group)

    # ✅ Duplicate Centre Midfielders into both Number 6 and Number 8
    cm_mask = df["Six-Group Position"].eq("Centre Midfield")
    if cm_mask.any():
        cm_as_6 = df.loc[cm_mask].copy()
        cm_as_8 = df.loc[cm_mask].copy()
        cm_as_6["Six-Group Position"] = "Number 6"
        cm_as_8["Six-Group Position"] = "Number 8"
        df = pd.concat([df, cm_as_6, cm_as_8], ignore_index=True)

    return df

# ============================================================
# Compute Scores (matches radar logic)
# ============================================================
def compute_scores(df_all: pd.DataFrame, min_minutes: int = 600) -> pd.DataFrame:
    pos_col = "Six-Group Position"
    df = df_all.copy()

    # --- Minutes setup ---
    df["Minutes played"] = pd.to_numeric(df.get("Minutes played", 0), errors="coerce").fillna(0)
    eligible = df[df["Minutes played"] >= min_minutes].copy()
    if eligible.empty:
        eligible = df.copy()

    # Ensure required columns exist
    for col in ["Avg Z Score", "Weighted Z Score", "LFC Weighted Z"]:
        if col not in df.columns:
            df[col] = 0.0
        if col not in eligible.columns:
            eligible[col] = 0.0

    # --- Step 1: Per-position Z-scores (baseline = all leagues) ---
    for position, template in position_metrics.items():
        metrics = template["metrics"]
        existing = [m for m in metrics if m in df.columns]
        if not existing:
            continue

        # Ensure metrics are numeric
        for m in existing:
            df[m] = pd.to_numeric(df[m], errors="coerce").fillna(0)

        elig_pos = eligible[eligible[pos_col] == position]
        if elig_pos.empty:
            continue

        means = elig_pos[existing].mean()
        stds = elig_pos[existing].std().replace(0, 1)

        mask = df[pos_col] == position
        Z = (df.loc[mask, existing] - means) / stds

        # Invert metrics where lower is better
        for m in (LOWER_IS_BETTER & set(existing)):
            Z[m] *= -1

        df.loc[mask, "Avg Z Score"] = Z.mean(axis=1).fillna(0)

    # --- Step 2: Weighted Z ---
    mult = pd.to_numeric(df.get("Multiplier", 1.0), errors="coerce").fillna(1.0)
    az = df["Avg Z Score"]
    df["Weighted Z Score"] = np.where(az > 0, az * mult, az / mult)

    # --- Step 3: LFC multiplier (1.20 for Scottish Premiership) ---
    df["LFC Multiplier"] = mult
    df.loc[df["Competition_norm"] == "Scotland Premiership", "LFC Multiplier"] = 1.20
    lfc_mult = df["LFC Multiplier"]
    df["LFC Weighted Z"] = np.where(az > 0, az * lfc_mult, az / lfc_mult)

    # --- Step 4: Rebuild eligible Weighted Z before anchors ---
    eligible = df[df["Minutes played"] >= min_minutes].copy()
    if eligible.empty:
        eligible = df.copy()

    if "Avg Z Score" not in eligible.columns:
        eligible["Avg Z Score"] = pd.to_numeric(eligible.get("Avg Z Score", 0), errors="coerce").fillna(0)
    eligible["Weighted Z Score"] = np.where(
        eligible["Avg Z Score"] > 0,
        eligible["Avg Z Score"] * pd.to_numeric(eligible["Multiplier"], errors="coerce").fillna(1.0),
        eligible["Avg Z Score"] / pd.to_numeric(eligible["Multiplier"], errors="coerce").fillna(1.0)
    )

    anchors = (
        eligible.groupby(pos_col, dropna=False)["Weighted Z Score"]
        .agg(_min="min", _max="max")
        .fillna(0)
    )
    df = df.merge(anchors, left_on=pos_col, right_index=True, how="left")

    # --- Step 5: Convert to 0–100 scale ---
    def to100(v, lo, hi):
        if pd.isna(v) or pd.isna(lo) or pd.isna(hi) or hi <= lo:
            return 50.0
        return np.clip((v - lo) / (hi - lo) * 100, 0.0, 100.0)

    df["Score (0–100)"] = [
        to100(v, lo, hi) for v, lo, hi in zip(df["Weighted Z Score"], df["_min"], df["_max"])
    ]
    df["LFC Score (0–100)"] = [
        to100(v, lo, hi) for v, lo, hi in zip(df["LFC Weighted Z"], df["_min"], df["_max"])
    ]

    df.drop(columns=["_min", "_max"], inplace=True, errors="ignore")
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

    df_team = df_all[(df_all[league_col] == selected_league) & (df_all["Team"] == selected_club)].copy()
    if df_team.empty:
        st.warning("No players found for this team.")
        st.stop()

    df_team["Minutes played"] = pd.to_numeric(df_team["Minutes played"], errors="coerce").fillna(0).astype(int)
    df_team["Rank in Team"] = df_team["Score (0–100)"].rank(ascending=False, method="min").astype(int)

    st.markdown("#### ⏱ Filter by Minutes Played (Display Only)")
    min_val, max_val = int(df_team["Minutes played"].min()), int(df_team["Minutes played"].max())
    selected_min_display = st.number_input(
        "Show only players with at least this many minutes",
        min_value=min_val, max_value=max_val,
        value=min(600, max_val), step=50
    )
    df_team = df_team[df_team["Minutes played"] >= selected_min_display]

    # --- Deduplicate players (keep the highest LFC Score row) ---
    if "Player" in df_team.columns and "LFC Score (0–100)" in df_team.columns:
        df_team = (
            df_team.sort_values("LFC Score (0–100)", ascending=False)
                   .drop_duplicates(subset=["Player"], keep="first")
                   .reset_index(drop=True)
        )

    avg_score = df_team["LFC Score (0–100)"].mean() if "LFC Score (0–100)" in df_team else np.nan
    st.markdown(f"### {selected_club} ({selected_league}) — Average {avg_score:.1f}" if not np.isnan(avg_score)
                else f"### {selected_club} ({selected_league}) — No eligible players")

    cols_for_table = [
        "Player", "Six-Group Position", "Team", league_col, "Multiplier",
        "Avg Z Score", "Weighted Z Score", "LFC Weighted Z",
        "Score (0–100)", "LFC Score (0–100)", "Age", "Minutes played", "Rank in Team"
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

    favs_in_db = {row[0] for row in get_favourites()}
    z_ranking["⭐ Favourite"] = z_ranking["Player"].isin(favs_in_db)

    st.data_editor(
        z_ranking,
        column_config={
            "⭐ Favourite": st.column_config.CheckboxColumn("⭐ Favourite"),
            "League Weight": st.column_config.NumberColumn("League Weight", format="%.3f"),
            "Z Avg": st.column_config.NumberColumn("Z Avg", format="%.3f"),
            "Z Weighted": st.column_config.NumberColumn("Z Weighted", format="%.3f"),
            "Score (0–100)": st.column_config.NumberColumn("Score (0–100)", format="%.1f"),
            "LFC Score (0–100)": st.column_config.NumberColumn("LFC Score (0–100)", format="%.1f"),
        },
        hide_index=False,
        width="stretch",
    )

except Exception as e:
    st.error(f"❌ Could not load data: {e}")
