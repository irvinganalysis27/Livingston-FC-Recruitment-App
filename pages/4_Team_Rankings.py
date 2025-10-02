import streamlit as st
import pandas as pd
from auth import check_password
from branding import show_branding

# ---------- Protect page ----------
if not check_password():
    st.stop()

# ---------- Branding ----------
show_branding()

# ---------- Page Title ----------
st.title("Team Rankings Page")
st.write("This page will show a team’s best XI by ranking and allow comparison of league-only vs whole dataset.")

# ---------- League & Club Filters ----------
# (Assumes df has Competition_norm and Team columns)
# Replace with your actual dataframe reference
try:
    league_options = sorted(df["Competition_norm"].dropna().unique())
    selected_league = st.selectbox("Select League", league_options)

    club_options = sorted(df.loc[df["Competition_norm"] == selected_league, "Team"].dropna().unique())
    selected_club = st.selectbox("Select Club", club_options)

    st.markdown(f"### Showing rankings for **{selected_club}** in {selected_league}")

    # Placeholder for formation output
    st.write("⚽ Formation (4-3-3) with ranked players will appear here.")

except Exception as e:
    st.warning("Data not yet loaded or missing required columns.")
    st.text(f"Error: {e}")
