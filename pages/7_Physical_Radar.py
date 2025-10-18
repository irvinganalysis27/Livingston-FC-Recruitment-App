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

    data = client.get_physical(params={
        'competition': '51,1385,4',  # Example: Scotland Premiership, Championship, League One
        'season': 2025,
        'group_by': 'player,team,competition,season,group',
        'playing_time__gte': 60,
        'count_match__gte': 5,
        'data_version': '3'
    })

    df = pd.DataFrame(data)
    df = p_utils.add_standard_metrics(df)

    # Save local backup (optional)
    df.to_csv("data/skillcorner_physical_backup.csv", index=False)

    print(f"[SkillCorner] Loaded {len(df)} rows at {datetime.now().strftime('%H:%M:%S')}")
    return df

# ========= RUN TEST =========
st.write("Fetching data from SkillCorner API...")
df_all_raw = load_skillcorner_data()

if df_all_raw.empty:
    st.error("❌ No data returned from SkillCorner API.")
    st.stop()

st.success(f"✅ Loaded {len(df_all_raw)} rows and {len(df_all_raw.columns)} columns.")
st.dataframe(df_all_raw.head(10))

# Show column names for analysis
st.markdown("### 🧩 Column names from API:")
st.write(df_all_raw.columns.tolist())
