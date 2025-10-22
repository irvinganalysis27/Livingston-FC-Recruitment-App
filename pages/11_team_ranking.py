import streamlit as st
import pandas as pd
from pages.1_Statsbomb_Radar import load_data_once, preprocess_df
from lib.scoring import compute_scores, position_metrics
from auth import check_password
from branding import show_branding

# --- Password gate ---
if not check_password():
    st.stop()

show_branding()
st.title("🏆 Team Rankings (Radar-based)")

# --- Load and compute ---
df_all_raw = load_data_once()
df_all = preprocess_df(df_all_raw)
df_all = compute_scores(df_all, min_minutes=600)

# --- Filters ---
league_col = "Competition_norm"
leagues = sorted(df_all[league_col].dropna().unique())
selected_league = st.selectbox("Select League", leagues)

clubs = sorted(df_all.loc[df_all[league_col] == selected_league, "Team"].dropna().unique())
selected_club = st.selectbox("Select Club", clubs)

templates = list(position_metrics.keys())
selected_template = st.selectbox("Select Position Template", templates)

# --- Filter dataset ---
df_team = df_all[
    (df_all[league_col] == selected_league)
    & (df_all["Team"] == selected_club)
    & (df_all["Six-Group Position"] == selected_template)
].copy()

if df_team.empty:
    st.warning("No players found for this position template.")
    st.stop()

# --- Display table ---
st.markdown(f"### {selected_club} — {selected_template}")
df_team = df_team.sort_values("Score (0–100)", ascending=False)

cols = ["Player", "Age", "Minutes played", "Score (0–100)", "LFC Score (0–100)"]
metric_cols = position_metrics[selected_template]["metrics"]
st.dataframe(df_team[cols + metric_cols], use_container_width=True)
