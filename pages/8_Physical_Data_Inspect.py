import streamlit as st
import pandas as pd

st.set_page_config(page_title="SkillCorner CSV Preview", layout="centered")
st.title("🧩 SkillCorner Physical Data Preview")

uploaded_path = "SkillCorner-2025-10-18.csv"  # ensure it’s in the app root

try:
    df = pd.read_csv(uploaded_path)
    st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns.")
    st.write("### Column Names:")
    st.write(df.columns.tolist())

    st.write("### First 10 Rows:")
    st.dataframe(df.head(10))
except Exception as e:
    st.error(f"❌ Failed to load: {e}")
