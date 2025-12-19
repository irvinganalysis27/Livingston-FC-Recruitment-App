import streamlit as st
import pandas as pd

from auth import check_password
from branding import show_branding
from ui.sidebar import render_sidebar

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
# Sidebar filters
# --------------------------------------------------
st.subheader("Filters")

col1, col2, col3 = st.columns(3)

with col1:
    position = st.selectbox(
        "Position",
        sorted(df["position"].dropna().unique())
    )

with col2:
    from_league = st.selectbox(
        "From league",
        sorted(df["from_league"].dropna().unique())
    )

with col3:
    to_league = st.selectbox(
        "To league",
        sorted(df["to_league"].dropna().unique())
    )

# --------------------------------------------------
# Filter data
# --------------------------------------------------
filt = df[
    (df["position"] == position) &
    (df["from_league"] == from_league) &
    (df["to_league"] == to_league)
]

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

A **{position}** moving from  
**{from_league} → {to_league}**

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
