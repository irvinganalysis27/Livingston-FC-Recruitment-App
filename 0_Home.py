import streamlit as st
from auth import check_password

st.set_page_config(page_title="Livingston FC Recruitment App", layout="wide")

# Protect this page too
if not check_password():
    st.stop()

st.title("Welcome Page")
st.markdown("✅ You are logged in. Use the sidebar to navigate.")
