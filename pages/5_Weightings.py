import streamlit as st
import pandas as pd
from auth import check_password
from branding import show_branding

# ---------- Protect page ----------
if not check_password():
    st.stop()

# ---------- Branding ----------
show_branding()

# ---------- Define weightings ----------
weightings_data = {
    "Australia A-League Men": 0.81,
    "Austria 2. Liga": 0.63,
    "Belgium Challenger Pro League": 0.78,
    "Bulgaria First League": 0.83,
    "Croatia 1. HNL": 1.07,
    "Czech First Tier": 1.00,
    "Denmark 1st Division": 0.62,
    "England League One": 0.97,
    "England League Two": 0.77,
    "England National League": 0.52,
    "Finland Veikkausliiga": 0.76,
    "France National 1": 0.79,
    "Germany 3. Liga": 0.81,
    "Hungary NB I": 0.98,
    "Iceland Besta Deild": 0.73,
    "Italy Serie C": 0.65,
    "Netherlands Eerste Divisie": 0.69,
    "Norway 1. Division": 0.72,
    "Poland 1 Liga": 0.74,
    "Poland Ekstraklasa": 1.09,
    "Republic of Ireland Premier Division": 0.81,
    "Romania Liga 1": 0.97,
    "Scotland Championship": 0.59,
    "Scotland Premiership": 1.00,
    "Serbia Super Liga": 0.84,
    "Slovakia 1. Liga": 0.89,
    "Slovenia 1. Liga": 0.87,
    "South Africa Premier Division": 0.86,
    "Sweden Superettan": 0.72,
    "Switzerland Challenge League": 0.71,
    "Tunisia Ligue 1": 0.72,
    "USA USL Championship": 0.61,
    "Portugal Liga 3": 0.64,
    "Portugal Liga Revelacao Sub 23": 0.64,
    "Canada Premier League": 0.61,
}

# Convert to DataFrame
df_weightings = pd.DataFrame(list(weightings_data.items()), columns=["League", "Multiplier"])

st.markdown("## League Weightings")

# Wrap table inside middle column to shrink width
left, mid, right = st.columns([1,2,1])  # adjust numbers to make it skinnier/wider
with mid:
    st.dataframe(df_weightings, use_container_width=True)

    # Optional: Download CSV button
    csv = df_weightings.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        csv,
        "league_weightings.csv",
        "text/csv",
        key="download-csv"
    )
