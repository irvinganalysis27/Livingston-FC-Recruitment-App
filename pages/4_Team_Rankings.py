import streamlit as st
import pandas as pd
from pathlib import Path
from auth import check_password
from branding import show_branding
import matplotlib.pyplot as plt

# ---------- Protect page ----------
if not check_password():
    st.stop()

# ---------- Branding ----------
show_branding()

# ---------- Page Title ----------
st.title("Team Rankings Page")
st.write("This page will show a team’s best XI by ranking and allow comparison of league-only vs whole dataset.")

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

try:
    df_all = load_data(DATA_PATH)

    # Make sure we have the normalised league column
    league_col = "Competition_norm" if "Competition_norm" in df_all.columns else "Competition"

    # ---------- League & Club Filters ----------
    league_options = sorted(df_all[league_col].dropna().unique())
    selected_league = st.selectbox("Select League", league_options)

    club_options = sorted(df_all.loc[df_all[league_col] == selected_league, "Team"].dropna().unique())
    selected_club = st.selectbox("Select Club", club_options)

    st.markdown(f"### Showing rankings for **{selected_club}** in {selected_league}")

    # ---------- Formation Plot Function ----------
    def plot_team_433(df, club_name):
        """
        Plot a 4-3-3 formation with ranked players for each position.
        df should already be filtered to the selected club.
        """
        # Define roles we want in formation
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

        # Grab best players in each role (ordered by Score or Rank if available)
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

        # --- Plot pitch (basic)
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.set_facecolor("green")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis("off")
        ax.set_title(f"{club_name} – Best XI (4-3-3)", color="white", fontsize=16, weight="bold")

        # Position coordinates (roughly placed)
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
            # Show the best player as circle
            ax.scatter(x, y, s=1200, c="white", edgecolors="black", zorder=3)
            # First player inside circle
            ax.text(x, y, players[0].split("(")[0], ha="center", va="center", fontsize=8, wrap=True)
            # Remaining players below
            if len(players) > 1:
                text_block = "\n".join(players[1:4])  # limit to top 3 extra
                ax.text(x, y - 7, text_block, ha="center", va="top", fontsize=6, color="yellow")

        st.pyplot(fig, use_container_width=True)

    # --- Run it for selected club ---
    df_club = df_all[df_all["Team"] == selected_club].copy()
    if not df_club.empty:
        plot_team_433(df_club, selected_club)
    else:
        st.warning("No players found for this club.")

except Exception as e:
    st.error(f"Could not load data. Error: {e}")
