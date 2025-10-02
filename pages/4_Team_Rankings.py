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
# Position mapping + metrics (now 8 groups)
# ============================================================

RAW_TO_EIGHT = {
    "RIGHTBACK": "Right Full Back", "RIGHTWINGBACK": "Right Full Back",
    "LEFTBACK": "Left Full Back", "LEFTWINGBACK": "Left Full Back",
    "RIGHTCENTREBACK": "Centre Back", "LEFTCENTREBACK": "Centre Back", "CENTREBACK": "Centre Back",
    "CENTREMIDFIELDER": "Centre Midfield",
    "RIGHTCENTREMIDFIELDER": "Centre Midfield", "LEFTCENTREMIDFIELDER": "Centre Midfield",
    "DEFENSIVEMIDFIELDER": "Number 6", "RIGHTDEFENSIVEMIDFIELDER": "Number 6", "LEFTDEFENSIVEMIDFIELDER": "Number 6",
    "CENTREDEFENSIVEMIDFIELDER": "Number 6",
    "CENTREATTACKINGMIDFIELDER": "Number 8", "ATTACKINGMIDFIELDER": "Number 8",
    "SECONDSTRIKER": "Number 8", "10": "Number 8",
    "RIGHTWING": "Right Winger", "RIGHTMIDFIELDER": "Right Winger",
    "LEFTWING": "Left Winger", "LEFTMIDFIELDER": "Left Winger",
    "CENTREFORWARD": "Striker", "RIGHTCENTREFORWARD": "Striker", "LEFTCENTREFORWARD": "Striker",
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

# Position metrics (same for left/right split)
base_fullback_metrics = [
    "Passing%","Pr. Pass% Dif.","Successful Crosses","Crossing%","Deep Progressions",
    "Successful Dribbles","Turnovers","OBV","Pass OBV",
    "Defensive Actions","Aerial Win%","PAdj Pressures",
    "PAdj Tack&Int","Dribbles Stopped%","Aggressive Actions","Player Season Ball Recoveries 90"
]

base_winger_metrics = [
    "xG","Shots","xG/Shot","Touches In Box","OP xG Assisted","NP Goals",
    "OP Passes Into Box","Successful Box Cross%","Passing%",
    "Successful Dribbles","Turnovers","OBV","D&C OBV","Fouls Won","Deep Progressions",
    "Player Season Fhalf Pressures 90"
]

position_metrics = {
    "Centre Back": {"metrics": ["NP Goals","Passing%","Pass OBV","Pr. Long Balls","UPr. Long Balls","OBV","Pr. Pass% Dif.","PAdj Interceptions","PAdj Tackles","Dribbles Stopped%","Defensive Actions","Aggressive Actions","Fouls","Aerial Wins","Aerial Win%"]},
    "Left Full Back": {"metrics": base_fullback_metrics},
    "Right Full Back": {"metrics": base_fullback_metrics},
    "Number 6": {"metrics": ["xGBuildup","xG Assisted","Passing%","Deep Progressions","Turnovers","OBV","Pass OBV","Pr. Pass% Dif.","PAdj Interceptions","PAdj Tackles","Dribbles Stopped%","Aggressive Actions","Aerial Win%","Player Season Ball Recoveries 90","Pressure Regains"]},
    "Number 8": {"metrics": ["xGBuildup","xG Assisted","Shots","xG","NP Goals","Passing%","Deep Progressions","OP Passes Into Box","Pass OBV","OBV","Deep Completions","Pressure Regains","PAdj Pressures","Player Season Fhalf Ball Recoveries 90","Aggressive Actions"]},
    "Left Winger": {"metrics": base_winger_metrics},
    "Right Winger": {"metrics": base_winger_metrics},
    "Striker": {"metrics": ["Aggressive Actions","NP Goals","xG","Shots","xG/Shot","Goal Conversion%","Touches In Box","xG Assisted","Fouls Won","Deep Completions","OP Key Passes","Aerial Win%","Aerial Wins","Player Season Fhalf Pressures 90"]},
    "Goalkeeper": {"metrics": []}
}

LOWER_IS_BETTER = {"Turnovers", "Fouls", "Pr. Long Balls", "UPr. Long Balls"}

# ============================================================
# Data loading
# ============================================================

def load_one_file(p: Path) -> pd.DataFrame:
    print(f"[DEBUG] Trying to load file at: {p.resolve()}")

    def try_excel() -> pd.DataFrame | None:
        try:
            import openpyxl
            return pd.read_excel(p, engine="openpyxl")
        except Exception:
            return None

    def try_csv() -> pd.DataFrame | None:
        for kwargs in [
            dict(sep=None, engine="python"),
            dict(),
            dict(encoding="latin1"),
        ]:
            try:
                return pd.read_csv(p, **kwargs)
            except Exception:
                continue
        return None

    df = None
    if p.suffix.lower() in {".xlsx", ".xls"}:
        df = try_excel()
        if df is None:
            df = try_csv()
    else:
        df = try_csv()
        if df is None:
            df = try_excel()

    if df is None:
        raise ValueError(f"Unsupported or unreadable file: {p.name}")

    print(f"[DEBUG] Loaded {p.name}, {len(df)} rows, {len(df.columns)} cols")
    return df

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
# Ranking
# ============================================================

def compute_rankings(df_all: pd.DataFrame, min_minutes: int = 600) -> pd.DataFrame:
    pos_col = "Six-Group Position"
    if pos_col not in df_all.columns:
        df_all[pos_col] = np.nan

    all_metrics = set()
    for v in position_metrics.values():
        all_metrics.update(v["metrics"])
    all_metrics = list(all_metrics)

    for m in all_metrics:
        if m not in df_all.columns:
            df_all[m] = 0
        df_all[m] = pd.to_numeric(df_all[m], errors="coerce").fillna(0)

    raw_z_all = pd.DataFrame(index=df_all.index, columns=all_metrics, dtype=float)
    for m in all_metrics:
        z_per_group = df_all.groupby(pos_col)[m].transform(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0)
        if m in LOWER_IS_BETTER:
            z_per_group *= -1
        raw_z_all[m] = z_per_group.fillna(0)

    df_all["Avg Z Score"] = raw_z_all.mean(axis=1).fillna(0)
    if "Multiplier" not in df_all.columns:
        df_all["Multiplier"] = 1.0
    df_all["Multiplier"] = pd.to_numeric(df_all["Multiplier"], errors="coerce").fillna(1.0)
    df_all["Weighted Z Score"] = df_all["Avg Z Score"] * df_all["Multiplier"]

    eligible = df_all[pd.to_numeric(df_all.get("Minutes played", np.nan), errors="coerce").fillna(0) >= min_minutes].copy()
    if eligible.empty:
        eligible = df_all.copy()

    anchor_minmax = eligible.groupby(pos_col)["Weighted Z Score"].agg(_scale_min="min", _scale_max="max").fillna(0)
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
# Formation plotting
# ============================================================

def plot_team_433(df, club_name, league_name):
    formation_roles = {
        "GK": ["Goalkeeper"],
        "LB": ["Left Full Back"], "LCB": ["Centre Back"], "RCB": ["Centre Back"], "RB": ["Right Full Back"],
        "CDM": ["Number 6"],
        "LCM": ["Number 8"], "RCM": ["Number 8"],
        "LW": ["Left Winger"], "ST": ["Striker"], "RW": ["Right Winger"],
    }

    team_players = {}
    used_players = set()
    for pos, roles in formation_roles.items():
        subset = df[df["Six-Group Position"].isin(roles)].copy()
        if "Score (0–100)" in subset.columns:
            subset = subset.sort_values("Score (0–100)", ascending=False)
        if not subset.empty:
            for _, r in subset.iterrows():
                if r["Player"] not in used_players:
                    team_players[pos] = [f"{r['Player']} ({r['Score (0–100)']:.0f})"]
                    used_players.add(r["Player"])
                    break
        else:
            team_players[pos] = ["-"]

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title(f"{club_name} ({league_name})", color="black", fontsize=16, weight="bold")

    coords = {
        "GK": (50, 5),
        "LB": (10, 25), "LCB": (37, 20), "RCB": (63, 20), "RB": (90, 25),
        "CDM": (50, 40),
        "LCM": (30, 55), "RCM": (70, 55),
        "LW": (15, 75), "ST": (50, 82), "RW": (85, 75),
    }

    for pos, (x, y) in coords.items():
        players = team_players.get(pos, ["-"])
        ax.text(x, y, players[0], ha="center", va="center", fontsize=9, color="black", weight="bold", wrap=True)

    st.pyplot(fig, use_container_width=True)

# ============================================================
# Main
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
