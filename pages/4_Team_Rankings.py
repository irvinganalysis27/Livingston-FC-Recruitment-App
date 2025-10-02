# pages/4_Team_Rankings.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from auth import check_password
from branding import show_branding
from datetime import datetime

# ---------- Protect page ----------
if not check_password():
    st.stop()

# ---------- Branding ----------
show_branding()

# ---------- Page Title ----------
st.title("Team Rankings Page")

# ---------- Load your data ----------
APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent
DATA_PATH = ROOT_DIR / "statsbomb_player_stats_clean.csv"
MULT_PATH = ROOT_DIR / "league_multipliers.xlsx"

# --- Helper: clean position strings ---
def _clean_pos_token(tok: str) -> str:
    if pd.isna(tok): return ""
    t = str(tok).upper().strip()
    t = re.sub(r"[.\-_/]", " ", t)
    t = re.sub(r"\s+", "", t)
    return t

RAW_TO_SIX = {
    "RIGHTBACK": "Full Back", "LEFTBACK": "Full Back",
    "RIGHTWINGBACK": "Full Back", "LEFTWINGBACK": "Full Back",
    "RIGHTCENTREBACK": "Centre Back", "LEFTCENTREBACK": "Centre Back", "CENTREBACK": "Centre Back",
    "CENTREMIDFIELDER": "Centre Midfield",
    "RIGHTCENTREMIDFIELDER": "Centre Midfield", "LEFTCENTREMIDFIELDER": "Centre Midfield",
    "DEFENSIVEMIDFIELDER": "Number 6", "RIGHTDEFENSIVEMIDFIELDER": "Number 6", "LEFTDEFENSIVEMIDFIELDER": "Number 6",
    "CENTREATTACKINGMIDFIELDER": "Number 8", "ATTACKINGMIDFIELDER": "Number 8",
    "SECONDSTRIKER": "Number 8", "10": "Number 8",
    "RIGHTWING": "Winger", "LEFTWING": "Winger",
    "RIGHTMIDFIELDER": "Winger", "LEFTMIDFIELDER": "Winger",
    "CENTREFORWARD": "Striker", "RIGHTCENTREFORWARD": "Striker", "LEFTCENTREFORWARD": "Striker",
    "GOALKEEPER": "Goalkeeper",
}

def map_first_position_to_group(primary_pos_cell) -> str:
    tok = _clean_pos_token(primary_pos_cell)
    return RAW_TO_SIX.get(tok, None)

# ---------- Load ----------
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    return df

try:
    df_all = load_data(DATA_PATH)

    # --- Clean headers ---
    df_all.columns = (
        df_all.columns.astype(str)
        .str.strip()
        .str.replace(u"\xa0", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )

    # --- Add age ---
    if "Birth Date" in df_all.columns:
        today = datetime.today()
        df_all["Age"] = pd.to_datetime(df_all["Birth Date"], errors="coerce").apply(
            lambda dob: today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            if pd.notna(dob) else np.nan
        )

    # --- Position mapping ---
    if "Position" in df_all.columns:
        df_all["Six-Group Position"] = df_all["Position"].apply(map_first_position_to_group)
    else:
        df_all["Six-Group Position"] = np.nan

    # --- Duplicate generic CMs into both 6 & 8 ---
    cm_mask = df_all["Six-Group Position"] == "Centre Midfield"
    if cm_mask.any():
        cm_rows = df_all.loc[cm_mask].copy()
        cm_as_6 = cm_rows.copy(); cm_as_6["Six-Group Position"] = "Number 6"
        cm_as_8 = cm_rows.copy(); cm_as_8["Six-Group Position"] = "Number 8"
        df_all = pd.concat([df_all, cm_as_6, cm_as_8], ignore_index=True)

    # --- League multipliers ---
    try:
        multipliers_df = pd.read_excel(MULT_PATH)
        if {"League", "Multiplier"}.issubset(multipliers_df.columns):
            df_all["Competition_norm"] = df_all["Competition"].astype(str).str.strip()
            df_all = df_all.merge(multipliers_df, left_on="Competition_norm", right_on="League", how="left")
            df_all["Multiplier"] = pd.to_numeric(df_all["Multiplier"], errors="coerce").fillna(1.0)
        else:
            st.warning("Multiplier file missing required columns. Using 1.0 for all.")
            df_all["Multiplier"] = 1.0
    except Exception as e:
        st.warning(f"Could not load multipliers: {e}. Using 1.0 for all.")
        df_all["Multiplier"] = 1.0

    # --- Ranking logic (Z + 0–100 Score) ---
    pos_col = "Six-Group Position"
    metric_cols = [c for c in df_all.columns if pd.api.types.is_numeric_dtype(df_all[c]) and c not in ["Age","Minutes played"]]

    raw_z_all = pd.DataFrame(index=df_all.index, columns=metric_cols, dtype=float)
    for m in metric_cols:
        group_stats = df_all.groupby(pos_col)[m].agg(['mean','std']).fillna(0)
        z_per_group = df_all.groupby(pos_col)[m].transform(lambda x: (x - x.mean())/x.std() if x.std()!=0 else 0)
        raw_z_all[m] = z_per_group.fillna(0)

    avg_z_all = raw_z_all.mean(axis=1)
    df_all["Avg Z Score"] = pd.to_numeric(avg_z_all, errors="coerce").fillna(0)
    df_all["Weighted Z Score"] = df_all["Avg Z Score"] * df_all["Multiplier"]

    anchor_minmax = (
        df_all.groupby(pos_col)["Weighted Z Score"]
              .agg(_scale_min="min", _scale_max="max")
              .fillna(0)
    )
    df_all = df_all.merge(anchor_minmax, left_on=pos_col, right_index=True, how="left")

    def _minmax_score(val, lo, hi):
        if hi <= lo: return 50.0
        return float(np.clip((val - lo) / (hi - lo) * 100.0, 0.0, 100.0))

    df_all["Score (0–100)"] = [
        _minmax_score(v, lo, hi)
        for v, lo, hi in zip(df_all["Weighted Z Score"], df_all["_scale_min"], df_all["_scale_max"])
    ]
    df_all["Score (0–100)"] = pd.to_numeric(df_all["Score (0–100)"], errors="coerce").round(1).fillna(0)
    df_all["Rank"] = df_all.groupby(pos_col)["Score (0–100)"].rank(ascending=False, method="min").astype(int)

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

        team_players = {}
        for pos, roles in formation_roles.items():
            subset = df[df["Six-Group Position"].isin(roles)].copy()
            if "Score (0–100)" in subset.columns:
                subset = subset.sort_values("Score (0–100)", ascending=False)
            if not subset.empty:
                players = [
                    f"{r['Player']} ({r['Score (0–100)']:.1f})"
                    for _, r in subset.iterrows()
                ]
                team_players[pos] = players
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
            "LB": (10, 20), "LCB": (37, 20), "RCB": (63, 20), "RB": (90, 20),
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

except Exception as e:
    st.error(f"Could not load data. Error: {e}")
