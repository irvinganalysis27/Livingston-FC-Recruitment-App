# streamlit_app.py
import streamlit as st
from auth import check_password

st.set_page_config(page_title="Livingston FC Recruitment App", layout="wide")

if not check_password():
    st.stop()

# Optional: redirect automatically to Home page
# st.switch_page("pages/0_Home.py")

st.title("Welcome Page")
st.markdown("✅ You are logged in. Use the sidebar to navigate.")
