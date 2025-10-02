# pages/4_Team_Rankings.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from datetime import datetime
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from auth import check_password
from branding import show_branding

# ============================================================
# Protect page
# ============================================================
if not check_password():
    st.stop()

show_branding()
st.title("Team Rankings Page")

APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent
DATA_PATH = ROOT_DIR / "statsbomb_player_stats_clean.csv"

# ============================================================
# Position mapping + metrics (same as radar page)
# ============================================================

RAW_TO_EIGHT = {
    # Full backs
    "LEFTBACK": "Left Back", "LEFTWINGBACK": "Left Back",
    "RIGHTBACK": "Right Back", "RIGHTWINGBACK": "Right Back",

    # Centre backs
    "CENTREBACK": "Centre Back",
    "LEFTCENTREBACK": "Centre Back",
    "RIGHTCENTREBACK": "Centre Back",

    # Defensive mids (Number 6)
    "DEFENSIVEMIDFIELDER": "Number 6",
    "LEFTDEFENSIVEMIDFIELDER": "Number 6",
    "RIGHTDEFENSIVEMIDFIELDER": "Number 6",
    "CENTREDEFENSIVEMIDFIELDER": "Number 6",

    # Central mids (Number 8)
    "CENTREMIDFIELDER": "Number 8",
    "LEFTCENTREMIDFIELDER": "Number 8",
    "RIGHTCENTREMIDFIELDER": "Number 8",

    # Attacking mids (Number 8)
    "CENTREATTACKINGMIDFIELDER": "Number 8",
    "LEFTATTACKINGMIDFIELDER": "Number 8",
    "RIGHTATTACKINGMIDFIELDER": "Number 8",
    "SECONDSTRIKER": "Number 8",
    "10": "Number 8",

    # Wingers
    "LEFTWING": "Left Wing",
    "LEFTMIDFIELDER": "Left Wing",
    "RIGHTWING": "Right Wing",
    "RIGHTMIDFIELDER": "Right Wing",

    # Strikers
    "CENTREFORWARD": "Striker",
    "LEFTCENTREFORWARD": "Striker",
    "RIGHTCENTREFORWARD": "Striker",

    # Goalkeeper
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
    return RAW_TO_EIGHT.get(tok, None)

# Position metrics from radar page
# ============================================================
# Position metrics (expanded to 8 groups)
# ============================================================

position_metrics = {
    "Centre Back": {
        "metrics": [
            "NP Goals", "Passing%", "Pass OBV", "Pr. Long Balls", "UPr. Long Balls",
            "OBV", "Pr. Pass% Dif.", "PAdj Interceptions", "PAdj Tackles",
            "Dribbles Stopped%", "Defensive Actions", "Aggressive Actions",
            "Fouls", "Aerial Wins", "Aerial Win%"
        ]
    },
    "Left Back": {
        "metrics": [
            "Passing%", "Pr. Pass% Dif.", "Successful Crosses", "Crossing%",
            "Deep Progressions", "Successful Dribbles", "Turnovers", "OBV", "Pass OBV",
            "Defensive Actions", "Aerial Win%", "PAdj Pressures",
            "PAdj Tack&Int", "Dribbles Stopped%", "Aggressive Actions",
            "Player Season Ball Recoveries 90"
        ]
    },
    "Right Back": {
        "metrics": [
            "Passing%", "Pr. Pass% Dif.", "Successful Crosses", "Crossing%",
            "Deep Progressions", "Successful Dribbles", "Turnovers", "OBV", "Pass OBV",
            "Defensive Actions", "Aerial Win%", "PAdj Pressures",
            "PAdj Tack&Int", "Dribbles Stopped%", "Aggressive Actions",
            "Player Season Ball Recoveries 90"
        ]
    },
    "Number 6": {
        "metrics": [
            "xGBuildup", "xG Assisted", "Passing%", "Deep Progressions", "Turnovers",
            "OBV", "Pass OBV", "Pr. Pass% Dif.", "PAdj Interceptions",
            "PAdj Tackles", "Dribbles Stopped%", "Aggressive Actions",
            "Aerial Win%", "Player Season Ball Recoveries 90", "Pressure Regains"
        ]
    },
    "Number 8": {
        "metrics": [
            "xGBuildup", "xG Assisted", "Shots", "xG", "NP Goals",
            "Passing%", "Deep Progressions", "OP Passes Into Box",
            "Pass OBV", "OBV", "Deep Completions", "Pressure Regains",
            "PAdj Pressures", "Player Season Fhalf Ball Recoveries 90",
            "Aggressive Actions"
        ]
    },
    "Left Wing": {
        "metrics": [
            "xG", "Shots", "xG/Shot", "Touches In Box", "OP xG Assisted", "NP Goals",
            "OP Passes Into Box", "Successful Box Cross%", "Passing%",
            "Successful Dribbles", "Turnovers", "OBV", "D&C OBV",
            "Fouls Won", "Deep Progressions", "Player Season Fhalf Pressures 90"
        ]
    },
    "Right Wing": {
        "metrics": [
            "xG", "Shots", "xG/Shot", "Touches In Box", "OP xG Assisted", "NP Goals",
            "OP Passes Into Box", "Successful Box Cross%", "Passing%",
            "Successful Dribbles", "Turnovers", "OBV", "D&C OBV",
            "Fouls Won", "Deep Progressions", "Player Season Fhalf Pressures 90"
        ]
    },
    "Striker": {
        "metrics": [
            "Aggressive Actions", "NP Goals", "xG", "Shots", "xG/Shot", "Goal Conversion%",
            "Touches In Box", "xG Assisted", "Fouls Won", "Deep Completions",
            "OP Key Passes", "Aerial Win%", "Aerial Wins", "Player Season Fhalf Pressures 90"
        ]
    },
    "Goalkeeper": {"metrics": []},  # keep for completeness
}

LOWER_IS_BETTER = {"Turnovers", "Fouls", "Pr. Long Balls", "UPr. Long Balls"}

# ============================================================
# Data loading + preprocessing
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

def pct_rank(series: pd.Series, lower_is_better: bool) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").fillna(0)
    r = series.rank(pct=True, ascending=True)
    if lower_is_better:
        p = 1.0 - r
    else:
        p = r
    return (p * 100.0).round(1)

# ============================================================
# Compute rankings (same logic, works with 8 groups)
# ============================================================

def compute_rankings(df_all: pd.DataFrame, min_minutes: int = 600) -> pd.DataFrame:
    pos_col = "Six-Group Position"
    if pos_col not in df_all.columns:
        df_all[pos_col] = np.nan

    # Collect all metrics across positions
    all_metrics = [m for v in position_metrics.values() for m in v["metrics"]]

    for m in all_metrics:
        if m not in df_all.columns:
            df_all[m] = 0
        df_all[m] = pd.to_numeric(df_all[m], errors="coerce").fillna(0)

    # Z-scores by position group
    raw_z_all = pd.DataFrame(index=df_all.index, columns=all_metrics, dtype=float)
    for m in all_metrics:
        z_per_group = df_all.groupby(pos_col)[m].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
        )
        if m in LOWER_IS_BETTER:
            z_per_group *= -1
        raw_z_all[m] = z_per_group.fillna(0)

    # Average Z-score
    df_all["Avg Z Score"] = raw_z_all.mean(axis=1).fillna(0)

    # Weighting with multipliers
    if "Multiplier" not in df_all.columns:
        df_all["Multiplier"] = 1.0
    df_all["Multiplier"] = pd.to_numeric(df_all["Multiplier"], errors="coerce").fillna(1.0)
    df_all["Weighted Z Score"] = df_all["Avg Z Score"] * df_all["Multiplier"]

    # Scale Weighted Z to 0–100 within each position group
    mins = pd.to_numeric(df_all.get("Minutes played", np.nan), errors="coerce").fillna(0)
    eligible = df_all[mins >= min_minutes].copy()
    if eligible.empty:
        eligible = df_all.copy()

    anchor_minmax = eligible.groupby(pos_col)["Weighted Z Score"].agg(
        _scale_min="min", _scale_max="max"
    ).fillna(0)

    df_all = df_all.merge(anchor_minmax, left_on=pos_col, right_index=True, how="left")

    def _minmax_score(v, lo, hi):
        if pd.isna(v) or pd.isna(lo) or pd.isna(hi) or hi <= lo:
            return 50.0
        return np.clip((v - lo) / (hi - lo) * 100.0, 0.0, 100.0)

    df_all["Score (0–100)"] = [
        _minmax_score(v, lo, hi)
        for v, lo, hi in zip(df_all["Weighted Z Score"], df_all["_scale_min"], df_all["_scale_max"])
    ]
    df_all["Score (0–100)"] = df_all["Score (0–100)"].round(1).fillna(0)

    # Ranking across all players
    df_all["Rank"] = df_all["Score (0–100)"].rank(ascending=False, method="min").astype(int)

    # Clean temp cols
    df_all.drop(columns=["_scale_min", "_scale_max"], inplace=True, errors="ignore")

    return df_all

# ============================================================
# Formation plotting (4-3-3 with 8 groups)
# ============================================================

def plot_team_433(df, club_name, league_name):
    # Define formation roles
    formation_roles = {
        "GK": ["Goalkeeper"],
        "LB": ["Left Back"],
        "RB": ["Right Back"],
        "LCB": ["Centre Back"],  # will pull best 2 CBs
        "RCB": ["Centre Back"],
        "CDM": ["Number 6"],
        "LCM": ["Number 8"],     # will pull best 2 8s
        "RCM": ["Number 8"],
        "LW": ["Left Wing"],
        "RW": ["Right Wing"],
        "ST": ["Striker"],
    }

    used_players = set()
    team_players = {}

    # Special handling for CBs (split into LCB + RCB)
    cb_subset = df[df["Six-Group Position"] == "Centre Back"].copy()
    cb_subset = cb_subset.sort_values("Score (0–100)", ascending=False)
    cb_players = cb_subset["Player"].tolist()
    cb_scores = cb_subset["Score (0–100)"].tolist()

    if len(cb_players) > 0:
        team_players["LCB"] = [f"{cb_players[0]} ({cb_scores[0]:.0f})"]
        used_players.add(cb_players[0])
    else:
        team_players["LCB"] = ["-"]

    if len(cb_players) > 1:
        team_players["RCB"] = [f"{cb_players[1]} ({cb_scores[1]:.0f})"]
        used_players.add(cb_players[1])
    else:
        team_players["RCB"] = ["-"]

    # Special handling for CMs (split into LCM + RCM)
    cm_subset = df[df["Six-Group Position"] == "Number 8"].copy()
    cm_subset = cm_subset.sort_values("Score (0–100)", ascending=False)
    cm_players = cm_subset["Player"].tolist()
    cm_scores = cm_subset["Score (0–100)"].tolist()

    if len(cm_players) > 0:
        team_players["LCM"] = [f"{cm_players[0]} ({cm_scores[0]:.0f})"]
        used_players.add(cm_players[0])
    else:
        team_players["LCM"] = ["-"]

    if len(cm_players) > 1:
        team_players["RCM"] = [f"{cm_players[1]} ({cm_scores[1]:.0f})"]
        used_players.add(cm_players[1])
    else:
        team_players["RCM"] = ["-"]

    # Fill other roles normally
    for pos, roles in formation_roles.items():
        if pos in ["LCB", "RCB", "LCM", "RCM"]:  # already handled
            continue

        subset = df[df["Six-Group Position"].isin(roles)].copy()
        subset = subset.sort_values("Score (0–100)", ascending=False)

        players = []
        for _, r in subset.iterrows():
            if r["Player"] not in used_players:
                players.append(f"{r['Player']} ({r['Score (0–100)']:.0f})")
                used_players.add(r["Player"])
            if len(players) >= 3:   # starter + 2 backups
                break

        team_players[pos] = players if players else ["-"]

    # Pitch plotting
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title(f"{club_name} ({league_name})", color="black", fontsize=16, weight="bold")

    # Coordinates for a 4-3-3
    coords = {
        "GK": (50, 5),
        "LB": (15, 25), "LCB": (35, 20), "RCB": (65, 20), "RB": (85, 25),
        "CDM": (50, 40),
        "LCM": (30, 55), "RCM": (70, 55),
        "LW": (20, 75), "ST": (50, 82), "RW": (80, 75),
    }

    # Place players neatly
    for pos, (x, y) in coords.items():
        players = team_players.get(pos, ["-"])
        # Starter bold at exact spot
        ax.text(x, y, players[0], ha="center", va="center",
                fontsize=9, color="black", weight="bold", wrap=True)
        # Backups stacked neatly below
        if len(players) > 1:
            backup_text = "\n".join(players[1:])
            ax.text(x, y - 5, backup_text, ha="center", va="top",
                    fontsize=7, color="black")

    st.pyplot(fig, use_container_width=True)

# ============================================================
# Main
# ============================================================

try:
    df_all_raw = load_statsbomb(DATA_PATH)
    df_all_raw = add_age_column(df_all_raw)

    # Preprocess
    df_all = df_all_raw.copy()
    if "Name" in df_all.columns:
        df_all.rename(columns={"Name": "Player"}, inplace=True)
    if "Primary Position" in df_all.columns:
        df_all.rename(columns={"Primary Position": "Position"}, inplace=True)
    if "Minutes" in df_all.columns:
        df_all.rename(columns={"Minutes": "Minutes played"}, inplace=True)

    # Map positions
    df_all["Six-Group Position"] = df_all["Position"].apply(map_first_position_to_group)

    # Compute rankings
    df_all = compute_rankings(df_all)

    # League filter
    league_col = "Competition_norm" if "Competition_norm" in df_all.columns else "Competition"
    league_options = sorted(df_all[league_col].dropna().unique())
    selected_league = st.selectbox("Select League", league_options)

    club_options = sorted(df_all.loc[df_all[league_col] == selected_league, "Team"].dropna().unique())
    selected_club = st.selectbox("Select Club", club_options)

    st.markdown(f"### Showing rankings for **{selected_club}** in {selected_league}")

    df_club = df_all[df_all["Team"] == selected_club].copy()
    plot_team_433(df_club, selected_club, selected_league)

except Exception as e:
    st.error(f"Could not load data. Error: {e}")
