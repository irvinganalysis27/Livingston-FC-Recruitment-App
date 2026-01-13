import streamlit as st
import pandas as pd
from auth import check_password
from branding import show_branding
from ui.sidebar import render_sidebar
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="League Transfer Analysis",
    layout="centered"
)

# --------------------------------------------------
# Auth
# --------------------------------------------------
if not check_password():
    st.stop()

render_sidebar()
show_branding()

st.title("League Transition Explorer")
st.caption(
    "Expected performance change when players move between leagues, "
    "based on historical player transitions using LFC scores."
)

# --------------------------------------------------
# Load data (cached)
# --------------------------------------------------
@st.cache_data
def load_transition_data():
    path = "data/league_transition_effects.csv"
    return pd.read_csv(path)

df = load_transition_data()

# --------------------------------------------------
# Filters
# --------------------------------------------------
st.subheader("Filters")

col1, col2, col3 = st.columns(3)

with col1:
    league_a = st.selectbox(
        "League A",
        sorted(pd.unique(pd.concat([df["from_league"], df["to_league"]]).dropna()))
    )

# Filter rows that involve League A
df_a = df[(df["from_league"] == league_a) | (df["to_league"] == league_a)]

with col2:
    league_b = st.selectbox(
        "League B",
        sorted(pd.unique(
            pd.concat([
                df_a["from_league"],
                df_a["to_league"]
            ]).dropna().unique())
        )
    )

# Filter rows that involve both leagues
df_ab = df_a[
    ((df_a["from_league"] == league_a) & (df_a["to_league"] == league_b)) |
    ((df_a["from_league"] == league_b) & (df_a["to_league"] == league_a))
]

with col3:
    position = st.selectbox(
        "Position",
        sorted(df_ab["position"].dropna().unique())
    )

# --------------------------------------------------
# Filter data
# --------------------------------------------------
filt = df_ab[df_ab["position"] == position]

# --------------------------------------------------
# Display result
# --------------------------------------------------
st.divider()

if filt.empty:
    st.warning("No historical transitions found for this combination.")
    st.stop()

row = filt.iloc[0]

avg_delta = row["avg_delta"]
median_delta = row["median_delta"]
pct_positive = row["pct_positive"]
n_players = int(row["n_players"])

# --------------------------------------------------
# Headline result
# --------------------------------------------------
delta_colour = "green" if avg_delta > 0 else "red" if avg_delta < 0 else "gray"
delta_sign = "+" if avg_delta > 0 else ""

st.markdown(
    f"""
### Expected outcome

A **{position}** moving between  
**{league_a} ↔ {league_b}**

is expected to change performance by:

<h2 style="color:{delta_colour};">
{delta_sign}{avg_delta:.1f}%
</h2>
""",
    unsafe_allow_html=True
)

# --------------------------------------------------
# Supporting stats
# --------------------------------------------------
c1, c2, c3 = st.columns(3)

c1.metric("Median change", f"{median_delta:+.1f}%")
c2.metric("Players improving", f"{pct_positive:.0f}%")
c3.metric("Sample size", n_players)

# --------------------------------------------------
# Confidence note
# --------------------------------------------------
if n_players < 5:
    st.warning(
        "⚠️ Low sample size. Use with caution."
    )
elif n_players < 10:
    st.info(
        "ℹ️ Moderate sample size."
    )
else:
    st.success(
        "✅ Strong sample size."
    )

# --------------------------------------------------
# Optional: show raw row
# --------------------------------------------------
with st.expander("Show raw data"):
    st.dataframe(filt, use_container_width=True)
