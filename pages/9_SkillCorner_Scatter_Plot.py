import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import re
from auth import check_password
from branding import show_branding

# ============================================================
# Authentication & Branding
# ============================================================
if not check_password():
    st.stop()

show_branding()
st.title("SkillCorner Scatter Plot")

# ============================================================
# Data Load
# ============================================================
ROOT_DIR = Path(__file__).parents[1]
DATA_PATH = ROOT_DIR / "SkillCorner-2025-10-18.csv"

@st.cache_data(ttl=86400)
def load_data(path: Path) -> pd.DataFrame:
    for sep in [",", "\t", None]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", on_bad_lines="skip")
            if df.shape[1] > 1:
                df.columns = df.columns.astype(str).str.strip()
                return df
        except Exception:
            continue
    raise ValueError("Unable to read CSV file.")

try:
    df_all = load_data(DATA_PATH)
except Exception as e:
    st.error(f"❌ Could not load SkillCorner data: {e}")
    st.stop()

if df_all.empty:
    st.error("❌ No data found.")
    st.stop()

# ============================================================
# Clean headers
# ============================================================
df_all.columns = (
    df_all.columns.astype(str)
    .str.strip()
    .str.replace(u"\xa0", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
)

# ============================================================
# Use SkillCorner's existing Position Group column (rename for consistency)
# ============================================================
if "Position Group" not in df_all.columns:
    st.error("❌ 'Position Group' column not found in SkillCorner dataset.")
    st.stop()

# Optional: normalise SkillCorner position group names to match your app
POSITION_RENAME = {
    "Central Defender": "Centre Back",
    "Center Forward": "Striker",
    "Wide Attacker": "Winger",
}
df_all["Position Group"] = df_all["Position Group"].replace(POSITION_RENAME)

# ============================================================
# Required columns
# ============================================================
league_col = "Competition"
minutes_col = "Minutes"
position_col = "Position Group"

if not all(c in df_all.columns for c in [league_col, minutes_col, position_col]):
    st.error("❌ Missing required columns in SkillCorner dataset.")
    st.stop()

# ============================================================
# Filters
# ============================================================
st.markdown("#### Filters")

# --- League Filter ---
all_leagues = sorted(df_all[league_col].dropna().unique().tolist())
if "scatt_sc_league_sel" not in st.session_state:
    st.session_state.scatt_sc_league_sel = all_leagues.copy()

b1, b2 = st.columns([1, 1])
with b1:
    if st.button("Select all leagues"):
        st.session_state.scatt_sc_league_sel = all_leagues.copy()
with b2:
    if st.button("Clear all leagues"):
        st.session_state.scatt_sc_league_sel = []

selected_leagues = st.multiselect(
    "Leagues",
    options=all_leagues,
    default=st.session_state.scatt_sc_league_sel,
    key="scatt_sc_league_sel",
    label_visibility="collapsed",
)

# --- Minutes Filter ---
min_minutes = st.number_input("Minimum minutes (≥)", min_value=0, value=200, step=50)

# --- Position Filter ---
pos_groups = sorted(df_all[position_col].dropna().unique().tolist())
selected_positions = st.multiselect(
    "Position Groups",
    options=pos_groups,
    default=pos_groups,
    key="scatt_sc_pos_sel",
)

# --- Apply Filters ---
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
    highlight_all = st.toggle("Colour by Position Group")
with c3:
    highlight_team = st.toggle("Highlight My Team")

my_team_name = "Livingston"

# ============================================================
# Compute averages & outliers
# ============================================================
x_mean = df[x_metric].mean()
y_mean = df[y_metric].mean()

df["_is_outlier"] = False
if highlight_outliers:
    x_std, y_std = df[x_metric].std(), df[y_metric].std()
    df["_is_outlier"] = (
        (abs(df[x_metric] - x_mean) > 2 * x_std)
        | (abs(df[y_metric] - y_mean) > 2 * y_std)
    )

# ============================================================
# Colour Mapping
# ============================================================
if highlight_all:
    color_col = position_col
    color_title = "Position Group"
else:
    color_col = None
    color_title = None

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
    opacity=0.8,
    height=650,
    title=f"{y_metric} vs {x_metric}",
)

# --- Average Lines ---
fig.add_shape(type="line", x0=x_mean, x1=x_mean, y0=df[y_metric].min(), y1=df[y_metric].max(),
              line=dict(color="black", dash="dot"))
fig.add_shape(type="line", x0=df[x_metric].min(), x1=df[x_metric].max(), y0=y_mean, y1=y_mean,
              line=dict(color="black", dash="dot"))

# --- Outliers ---
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

# --- Team Highlight (Gold for Livingston, no tooltip) ---
if highlight_team and "Team" in df.columns:
    team_mask = df["Team"].str.strip().eq(my_team_name)
    if team_mask.any():
        team_df = df[team_mask]

        fig.add_scatter(
            x=team_df[x_metric],
            y=team_df[y_metric],
            mode="markers",
            marker=dict(color="#FFD700", size=14, symbol="circle"),
            name=my_team_name,
            hoverinfo="skip",
            showlegend=True,
        )

        # fade other teams slightly (no “Other” legend)
        other_df = df[~team_mask]
        fig.add_scatter(
            x=other_df[x_metric],
            y=other_df[y_metric],
            mode="markers",
            marker=dict(color="rgba(120,180,170,0.5)", size=10),
            name="",
            hovertext=None,
            hoverinfo="none",
            showlegend=False,
        )

fig.update_layout(
    xaxis_title=x_metric,
    yaxis_title=y_metric,
    legend_title=color_title or "",
    template="plotly_white",
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Download CSV
# ============================================================
st.download_button(
    "📥 Download Filtered Data (CSV)",
    df.to_csv(index=False).encode("utf-8"),
    file_name="skillcorner_scatter_plot_data.csv",
    mime="text/csv",
)
