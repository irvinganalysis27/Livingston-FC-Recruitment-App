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
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    return pd.read_csv(p)

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
    elif "birth_date" in df.columns:
        today = datetime.today()
        df["Age"] = pd.to_datetime(df["birth_date"], errors="coerce").apply(
            lambda dob: today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            if pd.notna(dob) else np.nan
        )
    else:
        df["Age"] = np.nan
    return df

# ============================================================
# Position & League Mapping (IDENTICAL TO RADAR PAGE)
# ============================================================
def _clean_pos_token(tok: str) -> str:
    if pd.isna(tok):
        return ""
    t = str(tok).upper().strip()
    t = re.sub(r"[.\-_/]", " ", t)
    t = re.sub(r"\s+", "", t)
    return t

RAW_TO_SIX = {
    # Full backs & wing backs
    "RIGHTBACK": "Full Back", "LEFTBACK": "Full Back",
    "RIGHTWINGBACK": "Full Back", "LEFTWINGBACK": "Full Back",
    # Centre backs
    "RIGHTCENTREBACK": "Centre Back", "LEFTCENTREBACK": "Centre Back", "CENTREBACK": "Centre Back",
    # Centre mids
    "CENTREMIDFIELDER": "Centre Midfield", "RIGHTCENTREMIDFIELDER": "Centre Midfield", "LEFTCENTREMIDFIELDER": "Centre Midfield",
    # Defensive mids → 6
    "DEFENSIVEMIDFIELDER": "Number 6", "RIGHTDEFENSIVEMIDFIELDER": "Number 6",
    "LEFTDEFENSIVEMIDFIELDER": "Number 6", "CENTREDEFENSIVEMIDFIELDER": "Number 6",
    # Attacking mids → 8
    "CENTREATTACKINGMIDFIELDER": "Number 8", "ATTACKINGMIDFIELDER": "Number 8",
    "RIGHTATTACKINGMIDFIELDER": "Number 8", "LEFTATTACKINGMIDFIELDER": "Number 8",
    "SECONDSTRIKER": "Number 8", "10": "Number 8",
    # Wingers
    "RIGHTWING": "Winger", "LEFTWING": "Winger",
    "RIGHTMIDFIELDER": "Winger", "LEFTMIDFIELDER": "Winger",
    # Strikers
    "CENTREFORWARD": "Striker", "RIGHTCENTREFORWARD": "Striker", "LEFTCENTREFORWARD": "Striker",
    # Goalkeepers
    "GOALKEEPER": "Goalkeeper",
}

def map_first_position_to_group(primary_pos_cell) -> str:
    tok = _clean_pos_token(primary_pos_cell)
    return RAW_TO_SIX.get(tok, None)

LOWER_IS_BETTER = {"Turnovers", "Fouls", "Pr. Long Balls", "UPr. Long Balls"}

# ============================================================
# Preprocess (MATCHING RADAR PAGE)
# ============================================================
def preprocess_df(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    # League normalization
    if "Competition" in df.columns:
        df["Competition_norm"] = df["Competition"].astype(str).str.strip()
    else:
        df["Competition_norm"] = np.nan

    # Merge league multipliers
    try:
        m = pd.read_excel(ROOT_DIR / "league_multipliers.xlsx")
        m.columns = m.columns.str.lower().str.strip()
        if "league" in m.columns and "multiplier" in m.columns:
            df = df.merge(m, left_on="Competition_norm", right_on="league", how="left")
            df["Multiplier"] = pd.to_numeric(df["multiplier"], errors="coerce").fillna(1.0)
        else:
            df["Multiplier"] = 1.0
    except Exception:
        df["Multiplier"] = 1.0

    # Rename identifiers
    rename_map = {}
    if "Name" in df.columns: rename_map["Name"] = "Player"
    if "Primary Position" in df.columns: rename_map["Primary Position"] = "Position"
    if "Minutes" in df.columns: rename_map["Minutes"] = "Minutes played"
    df.rename(columns=rename_map, inplace=True)

    # Handle secondary position + combined string
    if "Position" in df.columns:
        if "Secondary Position" in df.columns:
            df["Positions played"] = df["Position"].fillna("").astype(str) + np.where(
                df["Secondary Position"].notna() & (df["Secondary Position"].astype(str) != ""),
                ", " + df["Secondary Position"].astype(str),
                ""
            )
        else:
            df["Positions played"] = df["Position"].astype(str)
    else:
        df["Positions played"] = np.nan

    # Map position group
    df["Six-Group Position"] = df["Position"].apply(map_first_position_to_group)

    # Duplicate generic "Centre Midfield" into both 6 & 8
    cm_mask = df["Six-Group Position"] == "Centre Midfield"
    if cm_mask.any():
        cm_rows = df.loc[cm_mask].copy()
        cm_as_6 = cm_rows.copy()
        cm_as_6["Six-Group Position"] = "Number 6"
        cm_as_8 = cm_rows.copy()
        cm_as_8["Six-Group Position"] = "Number 8"
        df = pd.concat([df, cm_as_6, cm_as_8], ignore_index=True)

    return df

# ============================================================
# Scoring Logic (MATCHING RADAR PAGE)
# ============================================================
def compute_scores(df_all: pd.DataFrame, min_minutes: int = 600) -> pd.DataFrame:
    df_all = df_all.copy()
    pos_col = "Six-Group Position"

    # Ensure numeric
    numeric_cols = [c for c in df_all.columns if pd.api.types.is_numeric_dtype(df_all[c])]
    df_all[numeric_cols] = df_all[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    # Compute Z per position
    df_num = df_all.select_dtypes(include=[np.number]).copy()
    metric_cols = [c for c in df_num.columns if c not in ["Age", "Height", "Minutes played", "Multiplier"]]
    raw_z = pd.DataFrame(index=df_all.index, columns=metric_cols, dtype=float)
    for m in metric_cols:
        z = df_all.groupby(pos_col)[m].transform(lambda x: (x - x.mean()) / x.std() if x.std() else 0)
        if m in LOWER_IS_BETTER:
            z *= -1
        raw_z[m] = z.fillna(0)

    df_all["Avg Z Score"] = raw_z.mean(axis=1).fillna(0)
    df_all["Weighted Z Score"] = df_all["Avg Z Score"] * pd.to_numeric(df_all["Multiplier"], errors="coerce").fillna(1.0)

    # Anchor scaling
    mins = pd.to_numeric(df_all.get("Minutes played", np.nan), errors="coerce").fillna(0)
    eligible = df_all[mins >= min_minutes].copy()
    if eligible.empty:
        eligible = df_all.copy()

    anchors = eligible.groupby(pos_col)["Weighted Z Score"].agg(_scale_min="min", _scale_max="max").fillna(0)
    df_all = df_all.merge(anchors, left_on=pos_col, right_index=True, how="left")

    def _to100(v, lo, hi):
        if pd.isna(v) or pd.isna(lo) or pd.isna(hi) or hi <= lo:
            return 50.0
        return np.clip((v - lo) / (hi - lo) * 100.0, 0.0, 100.0)

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
    df_raw = load_statsbomb(DATA_PATH)
    df_raw = add_age_column(df_raw)
    df_all = preprocess_df(df_raw)
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
    df_team["Minutes played"] = pd.to_numeric(df_team["Minutes played"], errors="coerce").fillna(0).astype(int)
    if "team_min_minutes" not in st.session_state:
        st.session_state.team_min_minutes = 600

    st.session_state.team_min_minutes = st.number_input(
        "Minimum minutes to include",
        min_value=0,
        max_value=int(df_team["Minutes played"].max()),
        value=st.session_state.team_min_minutes,
        step=50,
        key="team_min_minutes_input"
    )
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
        if c not in df_team.columns:
            df_team[c] = np.nan

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
