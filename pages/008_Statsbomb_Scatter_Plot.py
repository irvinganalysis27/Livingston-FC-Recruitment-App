# pages/9_Scatter_Plot.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import re
from auth import check_password
from branding import show_branding
from ui.sidebar import render_sidebar
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ============================================================
# Page Config & Authentication
# ============================================================
st.set_page_config(page_title="Livingston FC Recruitment App", layout="centered")


if not check_password():
    st.stop()
render_sidebar()

show_branding()
st.title("StatsBomb Scatter Plot")

# ============================================================
# Data Load
# ============================================================
ROOT_DIR = Path(__file__).parents[1]
DATA_PATH = ROOT_DIR / "statsbomb_player_stats_clean.csv"

@st.cache_data(ttl=86400)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    return df

try:
    df_all = load_data(DATA_PATH)
except Exception as e:
    st.error(f"❌ Could not load data: {e}")
    st.stop()

# --- Add Age column if birth date exists ---
if "Birth Date" in df_all.columns:
    today = pd.Timestamp.today()
    df_all["Age"] = pd.to_datetime(df_all["Birth Date"], errors="coerce").apply(
        lambda dob: today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        if pd.notna(dob) else np.nan
    )

if df_all.empty:
    st.error("❌ No data found.")
    st.stop()

# ============================================================
# Clean headers & drop irrelevant columns
# ============================================================
df_all.columns = (
    df_all.columns.astype(str)
    .str.strip()
    .str.replace(u"\xa0", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
)

drop_pattern = re.compile(r"(?i)(player season|account id)")
cols_to_drop = [c for c in df_all.columns if drop_pattern.search(c)]
if cols_to_drop:
    df_all.drop(columns=cols_to_drop, inplace=True, errors="ignore")

# ============================================================
# Position Mapping (Six-Group System)
# ============================================================
def _clean_pos_token(tok: str) -> str:
    if pd.isna(tok):
        return ""
    t = str(tok).upper().strip()
    t = re.sub(r"[.\-_/]", " ", t)
    t = re.sub(r"\s+", "", t)
    return t

RAW_TO_SIX = {
    "RIGHTBACK": "Full Back", "LEFTBACK": "Full Back",
    "RIGHTWINGBACK": "Full Back", "LEFTWINGBACK": "Full Back",
    "RIGHTCENTREBACK": "Centre Back", "LEFTCENTREBACK": "Centre Back", "CENTREBACK": "Centre Back",
    "DEFENSIVEMIDFIELDER": "Number 6", "RIGHTDEFENSIVEMIDFIELDER": "Number 6",
    "LEFTDEFENSIVEMIDFIELDER": "Number 6", "CENTREDEFENSIVEMIDFIELDER": "Number 6",
    "CENTREMIDFIELDER": "Centre Midfield", "RIGHTCENTREMIDFIELDER": "Centre Midfield", "LEFTCENTREMIDFIELDER": "Centre Midfield",
    "CENTREATTACKINGMIDFIELDER": "Number 8", "ATTACKINGMIDFIELDER": "Number 8",
    "RIGHTATTACKINGMIDFIELDER": "Number 8", "LEFTATTACKINGMIDFIELDER": "Number 8",
    "SECONDSTRIKER": "Number 8",
    "RIGHTWING": "Winger", "LEFTWING": "Winger",
    "RIGHTMIDFIELDER": "Winger", "LEFTMIDFIELDER": "Winger",
    "CENTREFORWARD": "Striker", "RIGHTCENTREFORWARD": "Striker", "LEFTCENTREFORWARD": "Striker",
}

def parse_first_position(cell):
    if pd.isna(cell):
        return ""
    return _clean_pos_token(str(cell))

def map_first_position_to_group(primary_pos_cell):
    tok = parse_first_position(primary_pos_cell)
    return RAW_TO_SIX.get(tok, None)

SIX_GROUPS = ["Full Back", "Centre Back", "Number 6", "Number 8", "Winger", "Striker"]

if "Primary Position" in df_all.columns:
    df_all["Six-Group Position"] = df_all["Primary Position"].apply(map_first_position_to_group)
else:
    df_all["Six-Group Position"] = np.nan

# Duplicate “Centre Midfield” rows into both Number 6 and Number 8
if "Six-Group Position" in df_all.columns:
    cm_mask = df_all["Six-Group Position"] == "Centre Midfield"
    if cm_mask.any():
        cm_rows = df_all.loc[cm_mask].copy()
        cm_as_6 = cm_rows.copy(); cm_as_6["Six-Group Position"] = "Number 6"
        cm_as_8 = cm_rows.copy(); cm_as_8["Six-Group Position"] = "Number 8"
        df_all = pd.concat([df_all, cm_as_6, cm_as_8], ignore_index=True)

# ============================================================
# Column validation
# ============================================================
league_col = "Competition"
minutes_col = "Minutes"
position_col = "Six-Group Position"

if league_col not in df_all.columns or minutes_col not in df_all.columns or position_col not in df_all.columns:
    st.error("❌ Could not find required columns in dataset.")
    st.stop()

# ============================================================
# Filters
# ============================================================
st.markdown("#### Filters")

# --- League Filter ---
all_leagues = sorted(df_all[league_col].dropna().unique().tolist())
if "sb_league_sel" not in st.session_state:
    st.session_state.sb_league_sel = all_leagues.copy()

b1, b2 = st.columns([1, 1])
with b1:
    if st.button("Select all leagues"):
        st.session_state.sb_league_sel = all_leagues.copy()
with b2:
    if st.button("Clear all leagues"):
        st.session_state.sb_league_sel = []

selected_leagues = st.multiselect(
    "Leagues",
    options=all_leagues,
    default=st.session_state.sb_league_sel,
    key="sb_league_sel",
    label_visibility="collapsed",
)

# --- Minutes Filter ---
min_minutes = st.number_input("Minimum minutes (≥)", min_value=0, value=600, step=50, key="sb_min_minutes")

# --- Age Filter ---
if "Age" in df_all.columns and df_all["Age"].notna().any():
    age_min = int(df_all["Age"].min())
    age_max = int(df_all["Age"].max())

    if "sb_age_range" not in st.session_state:
        st.session_state.sb_age_range = (age_min, age_max)

    sel_age_min, sel_age_max = st.slider(
        "Age range",
        min_value=age_min,
        max_value=age_max,
        value=st.session_state.sb_age_range,
        step=1,
        key="sb_age_range",
    )
else:
    sel_age_min, sel_age_max = None, None

# --- Position Filter ---
pos_groups = [g for g in SIX_GROUPS if g in df_all[position_col].unique()]
selected_positions = st.multiselect(
    "Position Groups",
    options=pos_groups,
    default=pos_groups,
    key="sb_pos_sel",
)

# --- Apply filters ---
df = df_all.copy()
if selected_leagues:
    df = df[df[league_col].isin(selected_leagues)]

df = df[pd.to_numeric(df[minutes_col], errors="coerce") >= min_minutes]

if sel_age_min is not None:
    df = df[df["Age"].between(sel_age_min, sel_age_max)]

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
x_metric = st.selectbox("X-axis metric", options=numeric_cols, index=0, key="sb_x_metric")
y_metric = st.selectbox("Y-axis metric", options=numeric_cols, index=1, key="sb_y_metric")

# ============================================================
# Toggles
# ============================================================
st.markdown("#### Highlight Options")
c1, c2, c3, c4 = st.columns(4)
with c1:
    highlight_outliers = st.toggle("Highlight Outliers", key="sb_outliers")
with c2:
    highlight_all = st.toggle("Colour by Position Group", key="sb_color_pos")
with c3:
    highlight_team = st.toggle("Highlight My Team", key="sb_team_highlight")
with c4:
    highlight_players = st.multiselect(
        "Highlight player(s)",
        options=sorted(df_all["Name"].dropna().unique().tolist()) if "Name" in df_all.columns else [],
        key="sb_player_highlight",
    )

my_team_name = "Livingston FC"

# ============================================================
# Averages & Outliers
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
    "Name": True if "Name" in df.columns else False,
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
        text=outliers["Name"] if "Name" in df.columns else None,
        textposition="top center",
        marker=dict(color="red", size=12, symbol="star"),
        name="Outliers",
    )

# --- Team Highlight (Gold, robust matching + team on tooltip) ---
if highlight_team and "Team" in df.columns:
    # Clean both sides and match on lowercase to handle 'Livingston' vs 'Livingston FC'
    df["_team_clean"] = df["Team"].astype(str).str.strip().str.lower()
    team_variants = ["livingston", "livingston fc"]

    team_mask = df["_team_clean"].isin(team_variants)
    if team_mask.any():
        team_df = df[team_mask]
        other_df = df[~team_mask]

        # Livingston FC (Gold)
        fig.add_scatter(
            x=team_df[x_metric],
            y=team_df[y_metric],
            mode="markers",
            marker=dict(color="#FFD700", size=14, symbol="circle"),
            name="Livingston FC",
            text=team_df["Name"] if "Name" in team_df.columns else team_df["Team"],
            customdata=np.stack(
                [team_df["Team"]], axis=-1
            ),  # pass Team column as custom data
            hovertemplate=(
                "%{text}<br>"
                "Team: %{customdata[0]}<br>"
                "%{xaxis.title.text}: %{x:.2f}<br>"
                "%{yaxis.title.text}: %{y:.2f}<extra></extra>"
            ),
            showlegend=True,
        )

        # Other players (keep hover info)
        fig.add_scatter(
            x=other_df[x_metric],
            y=other_df[y_metric],
            mode="markers",
            marker=dict(color="rgba(120,180,170,0.6)", size=10),
            name="Other Players",
            text=other_df["Name"] if "Name" in other_df.columns else other_df["Team"],
            customdata=np.stack(
                [other_df["Team"]], axis=-1
            ),
            hovertemplate=(
                "%{text}<br>"
                "Team: %{customdata[0]}<br>"
                "%{xaxis.title.text}: %{x:.2f}<br>"
                "%{yaxis.title.text}: %{y:.2f}<extra></extra>"
            ),
            showlegend=False,
        )

 # --- Specific Player Highlight ---
if "Name" in df.columns and "sb_player_highlight" in st.session_state:
    selected_players = st.session_state.sb_player_highlight
    if selected_players:
        player_mask = df["Name"].isin(selected_players)
        if player_mask.any():
            player_df = df[player_mask]

            fig.add_scatter(
                x=player_df[x_metric],
                y=player_df[y_metric],
                mode="markers+text",
                text=player_df["Name"],
                textposition="top center",
                marker=dict(
                    color="#E10600",  # strong red highlight
                    size=16,
                    symbol="diamond",
                    line=dict(color="black", width=1),
                ),
                name="Highlighted Players",
                customdata=np.stack(
                    [player_df["Team"]] if "Team" in player_df.columns else [player_df["Name"]],
                    axis=-1,
                ),
                hovertemplate=(
                    "%{text}<br>"
                    + ("Team: %{customdata[0]}<br>" if "Team" in player_df.columns else "")
                    + "%{xaxis.title.text}: %{x:.2f}<br>"
                    + "%{yaxis.title.text}: %{y:.2f}<extra></extra>"
                ),
                showlegend=True,
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
    file_name="statsbomb_scatter_plot_data.csv",
    mime="text/csv",
)
