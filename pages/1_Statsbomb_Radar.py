import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import re
from openai import OpenAI

# ========= APP CONFIG =========
st.set_page_config(page_title="Livingston FC Recruitment App", layout="centered")

# ========= AUTH / BRANDING =========
from auth import check_password
from branding import show_branding

# ========= SUPABASE + FAVOURITES =========
from lib.favourites_repo import (
    upsert_favourite,
    hide_favourite,
    list_favourites,
    get_supabase_client,
)

from lib.scoring import (
    preprocess_for_scoring,
    compute_scores,
)

# ========= PASSWORD GATE =========
if not check_password():
    st.stop()

show_branding()
st.title("StatsBomb Radar")

# ========= PATHS =========
APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent
ASSETS_DIR = ROOT_DIR / "assets"

# ========= OPENAI CLIENT =========
client = OpenAI(api_key=st.secrets["OpenAI"]["OPENAI_API_KEY"])

# ========= IMAGE HELPER =========
def open_image(path: Path):
    try:
        return Image.open(path)
    except Exception:
        return None

# ========= LOAD + PREPROCESS DATA =========
df_all = preprocess_for_scoring(ROOT_DIR)

if df_all is None or df_all.empty:
    st.error("❌ No player data loaded. Check your StatsBomb CSV path or contents.")
    st.stop()

# ========= LEAGUE FILTER =========
league_candidates = ["Competition_norm", "Competition", "competition_norm", "competition"]
league_col = next((c for c in league_candidates if c in df_all.columns), None)

if league_col is None:
    st.error("❌ No league/competition column found after preprocessing.")
    st.stop()

all_leagues = sorted(
    df_all[league_col].dropna().astype(str).str.strip().unique().tolist()
)

st.markdown("#### Choose league(s)")
if "league_selection" not in st.session_state:
    st.session_state.league_selection = all_leagues.copy()

b1, b2, _ = st.columns([1, 1, 6])
with b1:
    if st.button("Select all"):
        st.session_state.league_selection = all_leagues.copy()
with b2:
    if st.button("Clear all"):
        st.session_state.league_selection = []

selected_leagues = st.multiselect(
    "Leagues",
    options=all_leagues,
    default=st.session_state.league_selection,
    key="league_selection",
    label_visibility="collapsed",
)

if selected_leagues:
    df = df_all[df_all[league_col].isin(selected_leagues)].copy()
    st.caption(f"Leagues selected: {len(selected_leagues)} | Players: {len(df)}")
else:
    st.info("No leagues selected. Pick at least one or click ‘Select all’.")
    st.stop()

# ========= FILTERS: MINUTES + AGE =========
minutes_col = "Minutes played"
if minutes_col not in df.columns:
    df[minutes_col] = np.nan

c1, c2 = st.columns(2)
with c1:
    if "min_minutes" not in st.session_state:
        st.session_state.min_minutes = 500
    st.session_state.min_minutes = st.number_input(
        "Minimum minutes to include",
        min_value=0,
        value=st.session_state.min_minutes,
        step=50,
        key="min_minutes_input",
    )
    min_minutes = st.session_state.min_minutes
    df = df[pd.to_numeric(df[minutes_col], errors="coerce") >= min_minutes].copy()
    if df.empty:
        st.warning("No players meet the minutes threshold. Lower the minimum.")
        st.stop()

with c2:
    if "Age" in df.columns:
        df["_age_numeric"] = pd.to_numeric(df["Age"], errors="coerce")
        if df["_age_numeric"].notna().any():
            age_min, age_max = int(df["_age_numeric"].min()), int(df["_age_numeric"].max())
            if "age_range" not in st.session_state:
                st.session_state.age_range = (age_min, age_max)
            st.session_state.age_range = st.slider(
                "Age range to include",
                min_value=age_min,
                max_value=age_max,
                value=st.session_state.age_range,
                step=1,
                key="age_range_slider",
            )
            a_min, a_max = st.session_state.age_range
            df = df[df["_age_numeric"].between(a_min, a_max)].copy()

st.caption(f"Filtering on '{minutes_col}' ≥ {min_minutes}. Players remaining: {len(df)}")

# ========= POSITION GROUP SELECTION =========
SIX_GROUPS = ["Full Back", "Centre Back", "Number 6", "Number 8", "Winger", "Striker"]
st.markdown("#### 🟡 Select Position Group")

available_groups = [g for g in SIX_GROUPS if g in df["Six-Group Position"].unique()]
if "selected_groups" not in st.session_state:
    st.session_state.selected_groups = []

selected_groups = st.multiselect(
    "Position Groups",
    options=available_groups,
    default=st.session_state.selected_groups,
    label_visibility="collapsed",
)

if selected_groups:
    df = df[df["Six-Group Position"].isin(selected_groups)].copy()
    if df.empty:
        st.warning("No players after group filter.")
        st.stop()

current_single_group = selected_groups[0] if len(selected_groups) == 1 else None

# ========= TEMPLATE SELECT =========
st.markdown("#### 📊 Choose Radar Template")

template_names = list(position_metrics.keys())
if "template_select" not in st.session_state:
    st.session_state.template_select = template_names[0]

if current_single_group and current_single_group in position_metrics:
    st.session_state.template_select = current_single_group

selected_position_template = st.selectbox(
    "Radar Template",
    template_names,
    index=template_names.index(st.session_state.template_select),
    key="template_select",
    label_visibility="collapsed",
)

# ========= COMPUTE SCORING (from lib/scoring.py) =========
compute_within_league = st.checkbox(
    "Percentiles within each league", value=True, key="percentiles_within_league"
)

plot_data = compute_scores(
    df=df,
    df_all=df_all,
    selected_position_template=selected_position_template,
    compute_within_league=compute_within_league,
)

# ========= PLAYER SELECTION =========
players = plot_data["Player"].dropna().unique().tolist()
if not players:
    st.warning("No players found after filters.")
    st.stop()

if "selected_player" not in st.session_state:
    st.session_state.selected_player = players[0]

selected_player = st.selectbox(
    "Choose a player",
    players,
    index=players.index(st.session_state.selected_player)
    if st.session_state.selected_player in players
    else 0,
    key="player_select",
)
st.session_state.selected_player = selected_player

# ========= RADAR CHART =========
from lib.scoring import plot_radial_bar_grouped

if st.session_state.selected_player:
    plot_radial_bar_grouped(
        st.session_state.selected_player,
        plot_data,
        position_metrics[selected_position_template]["groups"],
        group_colors,
    )

# ========= AI SCOUTING SUMMARY =========
st.markdown("### 🧠 AI Scouting Summary")

def generate_player_summary(player_name, plot_data, metrics):
    try:
        row = plot_data.loc[plot_data["Player"] == player_name].iloc[0]
    except IndexError:
        return "No data available for this player."

    role = row.get("Six-Group Position", "player")
    league = row.get("Competition_norm", "")
    age = row.get("Age", "")
    team = row.get("Team", "")
    metric_percentiles = {
        m: row.get(f"{m} (percentile)", np.nan)
        for m in metrics.keys()
        if f"{m} (percentile)" in row.index
    }
    metric_text = ", ".join(
        [f"{k}: {v:.0f}" for k, v in metric_percentiles.items() if pd.notnull(v)]
    )

    prompt = f"""
    Write a concise scouting report on {player_name}, a {role.lower()} aged {age},
    currently in {league} for {team}. Metrics: {metric_text}.
    Be honest, analytical, and realistic in tone.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            temperature=0.85,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ AI summary generation failed: {e}"

if st.button("Generate AI Summary"):
    with st.spinner("Generating AI scouting report..."):
        st.markdown(
            generate_player_summary(
                st.session_state.selected_player,
                plot_data,
                position_metrics[selected_position_template]["groups"],
            )
        )

# ========= RANKING TABLE =========
st.markdown("### Player Rankings")
z_ranking = plot_data[
    [
        "Player",
        "Team",
        "Competition_norm",
        "Avg Z Score",
        "Weighted Z Score",
        "Score (0–100)",
        "Age",
        "Minutes played",
        "Rank",
    ]
].copy()
z_ranking.rename(columns={"Competition_norm": "League"}, inplace=True)
z_ranking.sort_values("Weighted Z Score", ascending=False, inplace=True, ignore_index=True)
z_ranking["Rank"] = np.arange(1, len(z_ranking) + 1)

st.dataframe(z_ranking, use_container_width=True)
