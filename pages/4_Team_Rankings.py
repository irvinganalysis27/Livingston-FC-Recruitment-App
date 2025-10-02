import streamlit as st
import pandas as pd
from pathlib import Path
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
st.write("This page will show a team’s best XI by ranking and allow comparison of league-only vs whole dataset.")

# ---------- Load your data ----------
ROOT_DIR = Path(__file__).parent.parent
DATA_PATH = ROOT_DIR / "statsbomb_player_stats_clean.csv"

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".csv"]:
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

    # ---------- Placeholder for formation ----------
    st.write("⚽ Formation (4-3-3) with ranked players will appear here.")

except Exception as e:
    st.error(f"Could not load data. Error: {e}")
