import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import re
from auth import check_password
from branding import show_branding

# ---------- Protect page ----------
if not check_password():
    st.stop()

# ---------- Branding ----------
show_branding()

# ---------- Page Title ----------
st.title("Team Rankings Page")

if DATA_PATH.exists():
    st.write(f"File found: {DATA_PATH}")
    st.write(f"File size: {DATA_PATH.stat().st_size} bytes")
else:
    st.error("File not found.")

# ---------- Load your data ----------
ROOT_DIR = Path(__file__).parent.parent
DATA_PATH = ROOT_DIR / "statsbomb_player_stats_clean.csv"
if not DATA_PATH.exists():
    st.error(f"Data file not found at {DATA_PATH}. Please upload or move it.")
    st.stop()

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    return df

# ---------- Position mapping helpers ----------
def _clean_pos_token(tok: str) -> str:
    if pd.isna(tok):
        return ""
    t = str(tok).upper().strip()
    t = re.sub(r"[.\-_/]", " ", t)
    t = re.sub(r"\s+", "", t)
    return t

RAW_TO_SIX = {
    # Full backs
    "RIGHTBACK": "Full Back", "LEFTBACK": "Full Back",
    "RIGHTWINGBACK": "Full Back", "LEFTWINGBACK": "Full Back",
    # Centre backs
    "RIGHTCENTREBACK": "Centre Back", "LEFTCENTREBACK": "Centre Back", "CENTREBACK": "Centre Back",
    # Midfielders
    "CENTREMIDFIELDER": "Centre Midfield",
    "RIGHTCENTREMIDFIELDER": "Centre Midfield", "LEFTCENTREMIDFIELDER": "Centre Midfield",
    "DEFENSIVEMIDFIELDER": "Number 6", "RIGHTDEFENSIVEMIDFIELDER": "Number 6", "LEFTDEFENSIVEMIDFIELDER": "Number 6",
    "CENTREATTACKINGMIDFIELDER": "Number 8", "ATTACKINGMIDFIELDER": "Number 8",
    "SECONDSTRIKER": "Number 8", "10": "Number 8",
    # Wingers
    "RIGHTWING": "Winger", "LEFTWING": "Winger",
    "RIGHTMIDFIELDER": "Winger", "LEFTMIDFIELDER": "Winger",
    # Strikers
    "CENTREFORWARD": "Striker", "RIGHTCENTREFORWARD": "Striker", "LEFTCENTREFORWARD": "Striker",
    # Goalkeeper
    "GOALKEEPER": "Goalkeeper",
}

def map_first_position_to_group(primary_pos_cell) -> str:
    tok = _clean_pos_token(primary_pos_cell)
    return RAW_TO_SIX.get(tok, None)

# ---------- Formation plotting ----------
def plot_team_433(df, club_name):
    """
    Plot a 4-3-3 formation with ranked players for each position.
    df should already be filtered to the selected club.
    """
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

    # Best players per role
    team_players = {}
    for pos, roles in formation_roles.items():
        subset = df[df["Six-Group Position"].isin(roles)].copy()
        if "Score (0–100)" in subset.columns:
            subset = subset.sort_values("Score (0–100)", ascending=False)
        elif "Rank" in subset.columns:
            subset = subset.sort_values("Rank", ascending=True)
        if not subset.empty:
            players = [f"{r['Player']} ({r.get('Score (0–100)', '')})" for _, r in subset.iterrows()]
            team_players[pos] = players
        else:
            team_players[pos] = ["-"]

    # --- Pitch
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_facecolor("green")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title(f"{club_name} – Best XI (4-3-3)", color="white", fontsize=16, weight="bold")

    # Position coords
    coords = {
        "GK": (50, 5),
        "LB": (20, 20), "LCB": (40, 20), "RCB": (60, 20), "RB": (80, 20),
        "CDM": (50, 40),
        "LCM": (30, 50), "RCM": (70, 50),
        "LW": (20, 75), "ST": (50, 80), "RW": (80, 75),
    }

    # Plot each role
    for pos, (x, y) in coords.items():
        players = team_players.get(pos, ["-"])
        ax.scatter(x, y, s=1200, c="white", edgecolors="black", zorder=3)
        ax.text(x, y, players[0].split("(")[0], ha="center", va="center", fontsize=8, wrap=True)
        if len(players) > 1:
            text_block = "\n".join(players[1:4])
            ax.text(x, y - 7, text_block, ha="center", va="top", fontsize=6, color="yellow")

    st.pyplot(fig, use_container_width=True)

# ---------- Main logic ----------
try:
    df_all = load_data(DATA_PATH)

    # Ensure Six-Group Position exists
    if "Position" in df_all.columns:
        df_all["Six-Group Position"] = df_all["Position"].apply(map_first_position_to_group)
    else:
        df_all["Six-Group Position"] = None

    # League filter
    league_col = "Competition_norm" if "Competition_norm" in df_all.columns else "Competition"
    league_options = sorted(df_all[league_col].dropna().unique())
    selected_league = st.selectbox("Select League", league_options)

    # Club filter
    club_options = sorted(df_all.loc[df_all[league_col] == selected_league, "Team"].dropna().unique())
    selected_club = st.selectbox("Select Club", club_options)

    st.markdown(f"### Showing rankings for **{selected_club}** in {selected_league}")

    # Filter to club + plot formation
    df_club = df_all[df_all["Team"] == selected_club].copy()
    plot_team_433(df_club, selected_club)

except Exception as e:
    st.error(f"Could not load data. Error: {e}")
