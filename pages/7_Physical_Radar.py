import streamlit as st
import pandas as pd
from datetime import datetime
from skillcorner.client import SkillcornerClient
from skillcornerviz.utils import skillcorner_physical_utils as p_utils

# ========= PAGE CONFIG =========
st.set_page_config(page_title="SkillCorner Physical Radar", layout="centered")
st.title("📊 SkillCorner Physical Data Test")

# ========= LOAD & CACHE =========
@st.cache_data(ttl=86400, show_spinner=True)
def load_skillcorner_data():
    client = SkillcornerClient(
        username=st.secrets["SKILLCORNER"]["USERNAME"],
        password=st.secrets["SKILLCORNER"]["PASSWORD"]
    )

    # --- Get available competitions ---
    try:
        comps = client.get_competitions()
        st.success(f"✅ Found {len(comps)} competitions available.")
        df_comps = pd.DataFrame(comps)
        st.dataframe(df_comps)
        return df_comps
    except Exception as e:
        st.error(f"❌ Failed to fetch competitions: {e}")
        return pd.DataFrame()

# ========= RUN TEST =========
st.write("Fetching data from SkillCorner API...")
df_all_raw = load_skillcorner_data()

st.markdown("### ✅ Raw columns returned from API:")
if df_all_raw.empty:
    st.error("❌ No data returned from SkillCorner API.")
else:
    st.success(f"✅ Loaded {len(df_all_raw)} rows and {len(df_all_raw.columns)} columns.")
    st.dataframe(df_all_raw.head(10))
    st.markdown("### 🧩 All column names:")
    st.code(", ".join(df_all_raw.columns.tolist()))
