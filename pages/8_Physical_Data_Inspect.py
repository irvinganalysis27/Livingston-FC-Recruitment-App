import streamlit as st
import pandas as pd

st.set_page_config(page_title="SkillCorner CSV Preview", layout="centered")
st.title("🧩 SkillCorner Physical Data Preview")

uploaded_path = "SkillCorner-2025-10-18.csv"  # make sure this path matches your upload

try:
    # More forgiving CSV reader
    df = pd.read_csv(uploaded_path, on_bad_lines='skip', engine='python')
    st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns (skipped any bad lines).")

    st.write("### Column Names:")
    st.write(df.columns.tolist())

    st.write("### First 10 Rows:")
    st.dataframe(df.head(10))
except Exception as e:
    st.error(f"❌ Failed to load CSV: {e}")
