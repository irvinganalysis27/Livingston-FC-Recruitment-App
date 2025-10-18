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

    # --- Minimal request: no filters ---
    try:
        data = client.get_physical(params={})
    except Exception as e:
        st.error(f"❌ API request failed: {e}")
        return pd.DataFrame()

    if not data:
        print("[DEBUG] Empty or null response from API.")
        return pd.DataFrame()

    print(f"[DEBUG] Type: {type(data)}, length: {len(data)}")

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        first_keys = list(data[0].keys())
        print(f"[DEBUG] Keys in first row: {first_keys}")
    else:
        print("[DEBUG] No valid dict-like rows in data.")

    df = pd.DataFrame(data)
    print(f"[DEBUG] Shape: {df.shape}")
    st.write(df.head())
    return df

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
