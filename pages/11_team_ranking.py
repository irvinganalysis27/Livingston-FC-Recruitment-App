import streamlit as st
import pandas as pd
from pages.1_Statsbomb_Radar import load_data_once, preprocess_df
from lib.scoring import compute_scores
from lib.metric_templates import position_metrics  # if defined separately

st.title("🏆 Team Rankings (Radar-based)")

# --- Load and process the shared dataset ---
df_all_raw = load_data_once()
df_all = preprocess_df(df_all_raw)
df_all = compute_scores(df_all, min_minutes=600)

# --- Dropdowns ---
league_col = "Competition_norm"
leagues = sorted(df_all[league_col].dropna().unique())
selected_league = st.selectbox("Select League", leagues)

clubs = sorted(df_all.loc[df_all[league_col] == selected_league, "Team"].dropna().unique())
selected_club = st.selectbox("Select Club", clubs)

templates = list(position_metrics.keys())
selected_template = st.selectbox("Select Position Template", templates)

# --- Filter data ---
df_team = df_all[
    (df_all[league_col] == selected_league)
    & (df_all["Team"] == selected_club)
    & (df_all["Six-Group Position"] == selected_template)
].copy()

if df_team.empty:
    st.warning("No players found for this position template.")
    st.stop()

# --- Sort and display ---
df_team.sort_values("Score (0–100)", ascending=False, inplace=True)

st.markdown(f"### {selected_club} — {selected_template}")
st.dataframe(
    df_team[
        ["Player", "Age", "Minutes played", "Score (0–100)", "LFC Score (0–100)"]
        + position_metrics[selected_template]["metrics"]
    ],
    use_container_width=True
)
