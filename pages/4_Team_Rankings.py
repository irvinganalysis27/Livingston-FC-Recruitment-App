# pages/4_Team_Rankings.py
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

from auth import check_password
from branding import show_branding
from data_loader import load_and_preprocess   # <- new unified loader

# ---------- Protect page ----------
if not check_password():
    st.stop()

# ---------- Branding ----------
show_branding()

# ---------- Page Title ----------
st.title("Team Rankings Page")

# ---------- Load your data ----------
ROOT_DIR = Path(__file__).parent.parent
DATA_PATH = ROOT_DIR / "statsbomb_player_stats_clean.csv"

df_all = load_and_preprocess(DATA_PATH)

# ---------- League & Club Filters ----------
league_col = "Competition_norm" if "Competition_norm" in df_all.columns else "Competition"
league_options = sorted(df_all[league_col].dropna().unique())
selected_league = st.selectbox("Select League", league_options)

club_options = sorted(df_all.loc[df_all[league_col] == selected_league, "Team"].dropna().unique())
selected_club = st.selectbox("Select Club", club_options)

st.markdown(f"### Showing rankings for **{selected_club}** in {selected_league}")

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

    # --- Pitch ---
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_facecolor("white")   # cleaner background
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title(f"{club_name} – Best XI (4-3-3)", color="black", fontsize=16, weight="bold")

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
        # Best player (main name)
        ax.text(x, y, players[0].split("(")[0], ha="center", va="center", fontsize=9, color="black", weight="bold", wrap=True)
        # Remaining players underneath
        if len(players) > 1:
            text_block = "\n".join(players[1:4])
            ax.text(x, y - 7, text_block, ha="center", va="top", fontsize=7, color="black")

    st.pyplot(fig, use_container_width=True)

# ---------- Filter + plot ----------
df_club = df_all[df_all["Team"] == selected_club].copy()
plot_team_433(df_club, selected_club)
