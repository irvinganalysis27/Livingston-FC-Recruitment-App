# pages/9_Scatter_Plot.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from auth import check_password
from branding import show_branding

# ============================================================
# Authentication & Branding
# ============================================================
if not check_password():
    st.stop()

show_branding()
st.title("📊 Metric Scatter Plot (Interactive)")

# ============================================================
# Data Load
# ============================================================
ROOT_DIR = Path(__file__).parents[1]
DATA_PATH = ROOT_DIR / "statsbomb_player_stats_clean.csv"

@st.cache_data(ttl=86400)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

try:
    df_all = load_data(DATA_PATH)
except Exception as e:
    st.error(f"❌ Could not load data: {e}")
    st.stop()

if df_all.empty:
    st.error("❌ No data found.")
    st.stop()

# ============================================================
# Identify column names safely
# ============================================================
# Find the best-matching competition column
possible_league_cols = ["Competition_norm", "Competition", "League", "Competition Name"]
league_col = next((c for c in possible_league_cols if c in df_all.columns), None)
if not league_col:
    st.error("❌ Could not find a competition/league column in the dataset.")
    st.stop()

# Fallbacks for minutes + position columns
minutes_col = next((c for c in ["Minutes played", "Minutes", "Mins"] if c in df_all.columns), None)
position_col = next((c for c in ["Six-Group Position", "Position Group Normalised", "Position"] if c in df_all.columns), None)

if not minutes_col or not position_col:
    st.error("❌ Could not find required 'minutes' or 'position' columns.")
    st.stop()

# ============================================================
# Filters (same style as radar page)
# ============================================================
st.markdown("#### Filters")

# --- League Filter ---
all_leagues = sorted(df_all[league_col].dropna().unique().tolist())
if "scatter_league_sel" not in st.session_state:
    st.session_state.scatter_league_sel = all_leagues.copy()

b1, b2 = st.columns([1, 1])
with b1:
    if st.button("Select all leagues"):
        st.session_state.scatter_league_sel = all_leagues.copy()
with b2:
    if st.button("Clear all leagues"):
        st.session_state.scatter_league_sel = []

selected_leagues = st.multiselect(
    "Leagues",
    options=all_leagues,
    default=st.session_state.scatter_league_sel,
    key="scatter_league_sel",
    label_visibility="collapsed",
)

# --- Minutes Filter ---
min_minutes = st.number_input("Minimum minutes (≥)", min_value=0, value=600, step=50)

# --- Position Filter ---
pos_groups = sorted(df_all[position_col].dropna().unique().tolist())
selected_positions = st.multiselect(
    "Position Groups",
    options=pos_groups,
    default=pos_groups,
    key="scatter_pos_sel",
)

# Apply filters
df = df_all.copy()
if selected_leagues:
    df = df[df[league_col].isin(selected_leagues)]
df = df[pd.to_numeric(df[minutes_col], errors="coerce") >= min_minutes]
if selected_positions:
    df = df[df[position_col].isin(selected_positions)]

if df.empty:
    st.warning("No players match selected filters.")
    st.stop()

# ============================================================
# Metric Selection
# ============================================================
numeric_cols = sorted(df.select_dtypes(include=[np.number]).columns.tolist())
if not numeric_cols:
    st.error("❌ No numeric columns found for plotting.")
    st.stop()

st.markdown("#### Choose metrics for X and Y axes")
x_metric = st.selectbox("X-axis metric", options=numeric_cols, index=0)
y_metric = st.selectbox("Y-axis metric", options=numeric_cols, index=1)

# ============================================================
# Toggles
# ============================================================
st.markdown("#### Highlight Options")
c1, c2, c3 = st.columns(3)
with c1:
    highlight_outliers = st.toggle("Highlight Outliers")
with c2:
    highlight_all = st.toggle("Highlight All")
with c3:
    highlight_team = st.toggle("Highlight My Team")

my_team_name = "Livingston FC"  # customise this!

# ============================================================
# Compute Sample Averages
# ============================================================
x_mean = df[x_metric].mean()
y_mean = df[y_metric].mean()

# ============================================================
# Outlier Detection
# ============================================================
df["_is_outlier"] = False
if highlight_outliers:
    x_std, y_std = df[x_metric].std(), df[y_metric].std()
    df["_is_outlier"] = (
        (abs(df[x_metric] - x_mean) > 2 * x_std)
        | (abs(df[y_metric] - y_mean) > 2 * y_std)
    )

# ============================================================
# Colour Mapping Logic
# ============================================================
if highlight_all:
    color_col = position_col
    color_title = "Position Group"
else:
    color_col = None
    color_title = None

if highlight_team and "Team" in df.columns:
    df["_color"] = np.where(df["Team"].eq(my_team_name), "My Team", "Other")
    color_col = "_color"
    color_title = "Team Highlight"

# ============================================================
# Plotly Scatter
# ============================================================
hover_data = {
    "Player": True if "Player" in df.columns else False,
    "Team": True if "Team" in df.columns else False,
    league_col: True,
    x_metric: ":.2f",
    y_metric: ":.2f",
}

fig = px.scatter(
    df,
    x=x_metric,
    y=y_metric,
    color=color_col,
    color_discrete_sequence=px.colors.qualitative.Set2,
    hover_data=hover_data,
    size=minutes_col if minutes_col in df.columns else None,
    opacity=0.75,
    height=650,
    title=f"{y_metric} vs {x_metric}",
)

# --- Add Sample Average Lines ---
fig.add_shape(type="line", x0=x_mean, x1=x_mean, y0=df[y_metric].min(), y1=df[y_metric].max(),
              line=dict(color="black", dash="dot"))
fig.add_shape(type="line", x0=df[x_metric].min(), x1=df[x_metric].max(), y0=y_mean, y1=y_mean,
              line=dict(color="black", dash="dot"))

# --- Outlier Highlight Overlay ---
if highlight_outliers:
    outliers = df[df["_is_outlier"]]
    fig.add_scatter(
        x=outliers[x_metric],
        y=outliers[y_metric],
        mode="markers+text",
        text=outliers["Player"] if "Player" in df.columns else None,
        textposition="top center",
        marker=dict(color="red", size=12, symbol="star"),
        name="Outliers",
    )

# --- Team Highlight Overlay ---
if highlight_team and "Team" in df.columns and my_team_name in df["Team"].values:
    team_df = df[df["Team"] == my_team_name]
    fig.add_scatter(
        x=team_df[x_metric],
        y=team_df[y_metric],
        mode="markers+text",
        text=team_df["Player"] if "Player" in df.columns else None,
        textposition="top center",
        marker=dict(color="green", size=10, symbol="circle"),
        name=my_team_name,
    )

fig.update_layout(
    xaxis_title=x_metric,
    yaxis_title=y_metric,
    legend_title=color_title or "",
    template="plotly_white",
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Download Filtered Data
# ============================================================
st.download_button(
    "📥 Download Filtered Data (CSV)",
    df.to_csv(index=False).encode("utf-8"),
    file_name="scatter_plot_data.csv",
    mime="text/csv",
)
