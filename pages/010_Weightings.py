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

st.set_page_config(page_title="Livingston FC Recruitment App", layout="centered")

# ---------- Authentication ----------
if not check_password():
    st.stop()

render_sidebar()

# ---------- Branding ----------
show_branding()
st.title("League Weightings")

# ============================================================
# ⚙️ League multipliers (with Competition_IDs)
# ============================================================
data = [
    (93, "Australia A-League Men", 0.81),
    (1581, "Austria 2. Liga", 0.63),
    (1035, "Belgium Challenger Pro League", 0.78),
    (1865, "Bulgaria First League", 0.83),
    (78, "Croatia 1. HNL", 1.07),
    (76, "Czech First Tier", 1.00),
    (260, "Denmark 1st Division", 0.62),
    (4, "England League One", 0.97),
    (5, "England League Two", 0.77),
    (65, "England National League", 0.52),
    (106, "Finland Veikkausliiga", 0.76),
    (129, "France National 1", 0.79),
    (179, "Germany 3. Liga", 0.81),
    (1522, "Hungary NB I", 0.98),
    (1607, "Iceland Besta Deild", 0.73),
    (1882, "Italy Serie C", 0.65),
    (63, "Netherlands Eerste Divisie", 0.69),
    (1442, "Norway 1. Division", 0.72),
    (1848, "Poland 1 Liga", 0.74),
    (38, "Poland Ekstraklasa", 1.09),
    (107, "Republic of Ireland Premier Division", 0.81),
    (349, "Romania Liga 1", 0.97),
    (1385, "Scotland Championship", 0.59),
    (51, "Scotland Premiership", 1.00),
    (79, "Serbia Super Liga", 0.84),
    (124, "Slovakia 1. Liga", 0.89),
    (1714, "Slovenia 1. Liga", 0.87),
    (229, "South Africa Premier Division", 0.86),
    (249, "Sweden Superettan", 0.72),
    (177, "Switzerland Challenge League", 0.71),
    (1239, "Tunisia Ligue 1", 0.72),
    (89, "USA USL Championship", 0.61),
    (1873, "Portugal Liga 3", 0.64),
    (1875, "Portugal Liga Revelacao Sub 23", 0.64),
    (303, "Canada Premier League", 0.61),
]

df_weightings = pd.DataFrame(data, columns=["Competition_ID", "League", "Multiplier"])

# ============================================================
# 📊 Display table (centered)
# ============================================================
left, mid, right = st.columns([1, 3, 1])
with mid:
    st.dataframe(df_weightings, use_container_width=True, height=750)

    # Save automatically as Excel for radar app
    df_weightings.to_excel("league_multipliers.xlsx", index=False)
    st.success("✅ league_multipliers.xlsx updated in project root.")

    # Optional CSV download
    csv = df_weightings.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download as CSV",
        data=csv,
        file_name="league_weightings.csv",
        mime="text/csv",
        key="download-csv"
    )

st.caption("These weightings are applied to adjust player Z-scores by league strength.")
