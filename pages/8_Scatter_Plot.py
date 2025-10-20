import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from auth import check_password
from branding import show_branding

# --- Authentication & branding ---
if not check_password():
    st.stop()

show_branding()
st.title("🔍 Metric Scatter Plot")

# --- Load data (reuse your df_all) ---
ROOT_DIR = Path(__file__).parents[1]
DATA_PATH = ROOT_DIR / "statsbomb_player_stats_clean.csv"

@st.cache_data(ttl=86400)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

df_all = load_data(DATA_PATH)

if df_all.empty:
    st.error("No data loaded.")
    st.stop()

# --- Filters: league, minutes, position (similar to radar page) ---
all_leagues = sorted(df_all["Competition_norm"].dropna().unique().tolist())
if "scatter_league_sel" not in st.session_state:
    st.session_state.scatter_league_sel = all_leagues.copy()

col1, col2 = st.columns(2)
with col1:
    if st.button("Select all leagues", key="scatter_sel_all"):
        st.session_state.scatter_league_sel = all_leagues.copy()
with col2:
    if st.button("Clear all leagues", key="scatter_clear_all"):
        st.session_state.scatter_league_sel = []

selected_leagues = st.multiselect(
    "Leagues",
    options=all_leagues,
    default=st.session_state.scatter_league_sel,
    key="scatter_league_sel",
)

min_minutes = st.number_input("Minimum minutes (≥)", min_value=0, value=600, step=50, key="scatter_min_minutes")

pos_groups = sorted(df_all["Six-Group Position"].dropna().unique().tolist())
selected_positions = st.multiselect(
    "Position Groups",
    options=pos_groups,
    default=pos_groups,
    key="scatter_pos_sel"
)

# --- Apply filters ---
df = df_all.copy()
if selected_leagues:
    df = df[df["Competition_norm"].isin(selected_leagues)]
df = df[df["Minutes played"] >= min_minutes]
if selected_positions:
    df = df[df["Six-Group Position"].isin(selected_positions)]

if df.empty:
    st.warning("No players match the selected filters.")
    st.stop()

# --- Metric selection for scatter plot ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
x_metric = st.selectbox("X-axis metric", options=numeric_cols, index=0, key="scatter_x")
y_metric = st.selectbox("Y-axis metric", options=numeric_cols, index=1, key="scatter_y")

# --- Plotting ---
fig, ax = plt.subplots(figsize=(10,6))
sc = ax.scatter(
    df[x_metric], df[y_metric],
    c="grey", alpha=0.6,
    s=20
)

ax.set_xlabel(x_metric)
ax.set_ylabel(y_metric)
ax.set_title(f"{y_metric} vs {x_metric}")

# Highlight selected candidate player (optional)
if "selected_player" in st.session_state:
    sel = st.session_state.selected_player
    if sel in df["Player"].values:
        sel_row = df[df["Player"] == sel].iloc[0]
        ax.scatter(
            sel_row[x_metric], sel_row[y_metric],
            c="red", s=80, edgecolors='black', label=sel
        )
        ax.legend()

st.pyplot(fig, use_container_width=True)

# --- Data download ---
csv = df[[ "Player", "Team", x_metric, y_metric ]].to_csv(index=False)
st.download_button("Download data (CSV)", csv, file_name="scatter_data.csv", mime="text/csv")
