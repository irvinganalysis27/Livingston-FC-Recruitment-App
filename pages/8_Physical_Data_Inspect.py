import streamlit as st
import pandas as pd

st.set_page_config(page_title="SkillCorner CSV Preview", layout="centered")
st.title("🧩 SkillCorner Physical Data Preview")

uploaded_path = "SkillCorner-2025-10-18.csv"  # ensure it's in the root folder

possible_separators = [",", ";", "\t", "|"]

for sep in possible_separators:
    try:
        df = pd.read_csv(
            uploaded_path,
            sep=sep,
            engine="python",
            encoding="utf-8-sig",
            on_bad_lines="skip"
        )
        if len(df.columns) > 1:
            st.success(f"✅ Loaded {len(df)} rows using separator '{sep}' — {len(df.columns)} columns.")
            st.write("### Column Names:")
            st.write(df.columns.tolist())
            st.write("### First 10 Rows:")
            st.dataframe(df.head(10))
            break
    except Exception as e:
        continue
else:
    st.error("❌ Could not parse the CSV with common separators (comma, semicolon, tab, pipe). Try re-exporting it.")
