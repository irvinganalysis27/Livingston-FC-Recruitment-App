# pages/4_Team_Rankings.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from datetime import datetime

from auth import check_password
from branding import show_branding
from data_loader import load_statsbomb, preprocess_df, _data_signature

# ---------- Protect page ----------
if not check_password():
    st.stop()

# ---------- Branding ----------
show_branding()

st.title("Team Rankings Page")

# ---------- Load your data ----------
APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent
DATA_PATH = ROOT_DIR / "statsbomb_player_stats_clean.csv"

df_all_raw = load_statsbomb(DATA_PATH, _sig=_data_signature(DATA_PATH))
df_all = preprocess_df(df_all_raw)

# ---------- Ranking logic (copied from radar page) ----------
pos_col = "Six-Group Position"
minutes_col = "Minutes played"

# Ensure numeric Minutes
if minutes_col not in df_all.columns:
    df_all[minutes_col] = np.nan
df_all[minutes_col] = pd.to_numeric(df_all[minutes_col], errors="coerce")

# Pick metrics for ranking = all numeric cols except identifiers
exclude_cols = {"Player", "Team", "Competition", "Competition_norm",
                "Six-Group Position", "Positions played", "Birth Date"}
sel_metrics = [c for c in df_all.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_all[c])]

LOWER_IS_BETTER = {"Turnovers", "Fouls", "Pr. Long Balls", "UPr. Long Balls"}

# Raw Z-scores per metric per position
raw_z_all = pd.DataFrame(index=df_all.index, columns=sel_metrics, dtype=float)
for m in sel_metrics:
    df_all[m] = pd.to_numeric(df_all[m], errors="coerce").fillna(0)
    z_per_group = df_all.groupby(pos_col)[m].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
    )
    if m in LOWER_IS_BETTER:
        z_per_group *= -1
    raw_z_all[m] = z_per_group.fillna(0)

# Average Z
df_all["Avg Z Score"] = raw_z_all.mean(axis=1)

# Weighted Z with multipliers
if "Multiplier" in df_all.columns:
    df_all["Multiplier"] = pd.to_numeric(df_all["Multiplier"], errors="coerce").fillna(1.0)
else:
    df_all["Multiplier"] = 1.0

df_all["Weighted Z Score"] = df_all["Avg Z Score"] * df_all["Multiplier"]

# Scale Weighted Z to 0–100 within each position group
anchor_minutes_floor = 600
eligible = df_all[df_all[minutes_col] >= anchor_minutes_floor].copy()
if eligible.empty:
    eligible = df_all.copy()

anchor_minmax = (
    eligible.groupby(pos_col)["Weighted Z Score"]
            .agg(_scale_min="min", _scale_max="max")
            .fillna(0)
)

df_all = df_all.merge(anchor_minmax, left_on=pos_col, right_index=True, how="left")

def _minmax_score(val, lo, hi):
    try:
        val, lo, hi = float(val), float(lo), float(hi)
    except Exception:
        return 0.0
    if hi <= lo:
        return 50.0
    return float(np.clip((val - lo) / (hi - lo) * 100.0, 0.0, 100.0))

df_all["Score (0–100)"] = [
    _minmax_score(v, lo, hi)
    for v, lo, hi in zip(df_all["Weighted Z Score"], df_all["_scale_min"], df_all["_scale_max"])
]
df_all["Score (0–100)"] = pd.to_numeric(df_all["Score (0–100)"], errors="coerce").round(1).fillna(0)

# Rank players within dataset
df_all["Rank"] = df_all["Score (0–100)"].rank(ascending=False, method="min").astype(int)

# ---------- League & Club Filters ----------
league_col = "Competition_norm" if "Competition_norm" in df_all.columns else "Competition"
league_options = sorted(df_all[league_col].dropna().unique())
selected_league = st.selectbox("Select League", league_options)

club_options = sorted(df_all.loc[df_all[league_col] == selected_league, "Team"].dropna().unique())
selected_club = st.selectbox("Select Club", club_options)

st.markdown(f"### Showing rankings for **{selected_club}** in {selected_league}")

# ---------- Formation plotting ----------
def plot_team_433(df, club_name, league_name):
    formation_roles = {
        "GK": ["Goalkeeper"],
        "LB": ["Full Back"],
        "LCB": ["Centre Back"],
        "RCB": ["Centre Back"],
        "RB": ["Full Back"],
        "CDM": ["Number 6"],
        "LCM": ["Number 8"],
        "RCM": ["Number 8"],
        "LW": ["Winger"],
        "RW": ["Winger"],
        "ST": ["Striker"],
    }

    score_col = "Score (0–100)"
    team_players = {}
    for pos, roles in formation_roles.items():
        subset = df[df["Six-Group Position"].isin(roles)].copy()
        if score_col in subset.columns:
            subset = subset.sort_values(score_col, ascending=False)
        elif "Rank" in subset.columns:
            subset = subset.sort_values("Rank", ascending=True)

        if not subset.empty:
            players = [
                f"{r['Player']} ({int(round(r.get(score_col, 0)))})"
                for _, r in subset.iterrows()
            ]
            team_players[pos] = players
        else:
            team_players[pos] = ["-"]

    # --- Pitch
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
        ax.text(x, y, players[0], ha="center", va="center",
                fontsize=9, color="black", weight="bold", wrap=True)
        if len(players) > 1:
            text_block = "\n".join(players[1:4])
            ax.text(x, y - 3, text_block, ha="center", va="top",
                    fontsize=7, color="black")

    st.pyplot(fig, use_container_width=True)

# ---------- Filter + plot ----------
df_club = df_all[df_all["Team"] == selected_club].copy()
plot_team_433(df_club, selected_club, selected_league)
