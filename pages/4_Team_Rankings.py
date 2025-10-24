import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime, timezone
import time

from auth import check_password
from branding import show_branding
from supabase import create_client
from lib.favourites_repo import upsert_favourite, hide_favourite, list_favourites, get_supabase_client

st.set_page_config(page_title="Livingston FC Recruitment App", layout="centered")

# ---------- Authentication ----------
if not check_password():
    st.stop()

# ---------- Branding ----------
show_branding()
st.title("Team Player Rankings")

# ============================================================
# Paths & setup
# ============================================================
APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent
DATA_PATH = ROOT_DIR / "statsbomb_player_stats_clean.csv"

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

    try:
        mult = pd.read_excel(ROOT_DIR / "league_multipliers.xlsx")
        if {"League", "Multiplier"}.issubset(mult.columns):
            df = df.merge(mult, left_on="Competition_norm", right_on="League", how="left")
            df["Multiplier"] = pd.to_numeric(df["Multiplier"], errors="coerce").fillna(1.0)
        else:
            df["Multiplier"] = 1.0
    except Exception:
        df["Multiplier"] = 1.0

    df["Six-Group Position"] = df["Position"].apply(map_first_position_to_group)
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
    df["Minutes played"] = pd.to_numeric(df.get("Minutes played", 0), errors="coerce").fillna(0)
    eligible = df[df["Minutes played"] >= min_minutes].copy()
    if eligible.empty:
        eligible = df.copy()

    for col in ["Avg Z Score", "Weighted Z Score", "LFC Weighted Z"]:
        if col not in df.columns:
            df[col] = 0.0

    for position, template in position_metrics.items():
        metrics = template["metrics"]
        existing = [m for m in metrics if m in df.columns]
        if not existing:
            continue

        for m in existing:
            df[m] = pd.to_numeric(df[m], errors="coerce").fillna(0)

        elig_pos = eligible[eligible[pos_col] == position]
        if elig_pos.empty:
            continue

        means = elig_pos[existing].mean()
        stds = elig_pos[existing].std().replace(0, 1)
        mask = df[pos_col] == position
        Z = (df.loc[mask, existing] - means) / stds
        for m in (LOWER_IS_BETTER & set(existing)):
            Z[m] *= -1
        df.loc[mask, "Avg Z Score"] = Z.mean(axis=1).fillna(0)

    mult = pd.to_numeric(df.get("Multiplier", 1.0), errors="coerce").fillna(1.0)
    az = df["Avg Z Score"]
    df["Weighted Z Score"] = np.where(az > 0, az * mult, az / mult)

    df["LFC Multiplier"] = mult
    df.loc[df["Competition_norm"] == "Scotland Premiership", "LFC Multiplier"] = 1.20
    lfc_mult = df["LFC Multiplier"]
    df["LFC Weighted Z"] = np.where(az > 0, az * lfc_mult, az / lfc_mult)

    anchors = (
        df.groupby(pos_col, dropna=False)["Weighted Z Score"]
        .agg(_min="min", _max="max")
        .fillna(0)
    )
    df = df.merge(anchors, left_on=pos_col, right_index=True, how="left")

    def to100(v, lo, hi):
        if pd.isna(v) or pd.isna(lo) or pd.isna(hi) or hi <= lo:
            return 50.0
        return np.clip((v - lo) / (hi - lo) * 100, 0.0, 100.0)

    df["Score (0–100)"] = [to100(v, lo, hi) for v, lo, hi in zip(df["Weighted Z Score"], df["_min"], df["_max"])]
    df["LFC Score (0–100)"] = [to100(v, lo, hi) for v, lo, hi in zip(df["LFC Weighted Z"], df["_min"], df["_max"])]

    df.drop(columns=["_min", "_max"], inplace=True, errors="ignore")
    return df

# ============================================================
# Load & Filter
# ============================================================
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

# ============================================================
# Supabase Favourites (identical to radar)
# ============================================================
def get_favourites_with_colours_live():
    sb = get_supabase_client()
    if sb is None:
        return {}
    try:
        res = sb.table("favourites").select("*").execute()
        if not res.data:
            return {}
        return {
            r.get("player"): {
                "colour": r.get("colour", ""),
                "comment": r.get("comment", ""),
                "visible": bool(r.get("visible", True)),
            }
            for r in res.data
            if r.get("player")
        }
    except Exception as e:
        st.warning(f"⚠️ Could not load favourites: {e}")
        return {}

COLOUR_EMOJI = {
    "🟣 Needs Checked": "🟣", "🟡 Monitor": "🟡", "🟢 Go": "🟢", "🟠 Out Of Reach": "🟠", "🔴 No Further Interest": "🔴",
    "Needs Checked": "🟣", "Monitor": "🟡", "Go": "🟢", "No Further Interest": "🔴",
    "🟣": "🟣", "🟡": "🟡", "🟢": "🟢", "🟠": "🟠", "🔴": "🔴",
}

def colourize_player_name(name: str, favs_dict: dict) -> str:
    data = favs_dict.get(name)
    if not data:
        return name
    emoji = COLOUR_EMOJI.get(str(data.get("colour", "")).strip(), "")
    return f"{emoji} {name}" if emoji else name

# ============================================================
# Display Table
# ============================================================
favs = get_favourites_with_colours_live()
df_team["Player (coloured)"] = df_team["Player"].apply(lambda n: colourize_player_name(n, favs))
df_team["⭐ Favourite"] = df_team["Player"].apply(lambda n: bool(favs.get(n, {}).get("visible", False)))

sig_parts = (selected_league, selected_club, len(df_team))
editor_key = f"team_rankings_editor_{hash(sig_parts)}"

edited_df = st.data_editor(
    df_team[
        ["⭐ Favourite", "Player (coloured)", "Six-Group Position", "Team", "Competition_norm",
         "Multiplier", "Avg Z Score", "Weighted Z Score", "LFC Weighted Z",
         "Score (0–100)", "LFC Score (0–100)", "Age", "Minutes played", "Rank in Team"]
    ],
    column_config={
        "Player (coloured)": st.column_config.TextColumn("Player"),
        "⭐ Favourite": st.column_config.CheckboxColumn("⭐ Favourite", help="Syncs to Supabase"),
        "Multiplier": st.column_config.NumberColumn("League Weight", format="%.3f"),
        "Avg Z Score": st.column_config.NumberColumn("Z Avg", format="%.3f"),
        "Weighted Z Score": st.column_config.NumberColumn("Z Weighted", format="%.3f"),
        "LFC Score (0–100)": st.column_config.NumberColumn("LFC Score (0–100)", format="%.1f"),
    },
    hide_index=False,
    width="stretch",
    key=editor_key,
)

# ============================================================
# Sync with Supabase (identical to radar)
# ============================================================
@st.cache_data(ttl=5, show_spinner=False)
def load_favourites_cached():
    return get_favourites_with_colours_live()

favs_live = load_favourites_cached()

if not st.session_state.get("_last_sync_time") or time.time() - st.session_state["_last_sync_time"] >= 3:
    st.session_state["_last_sync_time"] = time.time()

    if "⭐ Favourite" not in edited_df.columns:
        st.warning("⚠️ Could not find '⭐ Favourite' column — skipping sync.")
    else:
        favourite_rows = edited_df[edited_df["⭐ Favourite"] == True].copy()
        deleted_players = {p for p, d in favs_live.items() if not d.get("visible", True)}

        for _, row in favourite_rows.iterrows():
            player_raw = str(row.get("Player (coloured)", "")).strip()
            player_name = re.sub(r"^[🟢🟡🔴🟣🟠]\s*", "", player_raw).strip()
            team = row.get("Team", "")
            league = row.get("Competition_norm", "")
            position = row.get("Six-Group Position", "")

            prev_data = favs_live.get(player_name, {})
            prev_visible = bool(prev_data.get("visible", False))

            if player_name in deleted_players and not prev_visible:
                continue

            if not prev_visible:
                payload = {
                    "player": player_name,
                    "team": team,
                    "league": league,
                    "position": position,
                    "colour": prev_data.get("colour", "🟣 Needs Checked"),
                    "comment": prev_data.get("comment", ""),
                    "visible": True,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "source": "team-page",
                }
                upsert_favourite(payload, log_to_sheet=True)

        non_fav_rows = edited_df[edited_df["⭐ Favourite"] == False]
        for _, row in non_fav_rows.iterrows():
            player_raw = str(row.get("Player (coloured)", "")).strip()
            player_name = re.sub(r"^[🟢🟡🔴🟣🟠]\s*", "", player_raw).strip()
            old_visible = favs_live.get(player_name, {}).get("visible", False)
            if old_visible:
                hide_f
