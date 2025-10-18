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

    # Fetch raw data
    data = client.get_physical(params={
        'competition': '51,1385,4',  # Example: Scotland Premiership, Championship, League One
        'season': 2025,
        'group_by': 'player,team,competition,season,group',
        'playing_time__gte': 60,
        'count_match__gte': 5,
        'data_version': '3'
    })

    # Check raw type and content
    print(f"[DEBUG] Type of data returned: {type(data)}")
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        first_row_keys = list(data[0].keys())
        print(f"[DEBUG] Keys in first row: {first_row_keys}")
    else:
        print("[DEBUG] No valid dict-like rows found.")

    df = pd.DataFrame(data)
    print(f"[DEBUG] DataFrame shape: {df.shape}")

    # Try adding metrics (safe)
    try:
        df = p_utils.add_standard_metrics(df)
        print("[DEBUG] Added SkillCorner standard metrics successfully.")
    except Exception as e:
        print(f"[DEBUG] ⚠️ Could not add standard metrics: {e}")

    # Skip saving to non-existent folder
    try:
        df.to_csv("skillcorner_physical_backup.csv", index=False)
    except Exception as e:
        print(f"[DEBUG] ⚠️ Backup save failed: {e}")

    print(f"[SkillCorner] Loaded {len(df)} rows at {datetime.now().strftime('%H:%M:%S')}")
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
