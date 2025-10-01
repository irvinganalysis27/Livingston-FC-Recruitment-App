import streamlit as st
from auth import check_password
from branding import show_branding

st.set_page_config(page_title="Livingston FC Recruitment App", layout="wide")

# Protect page
from auth import check_password
if not check_password():
    st.stop()

# Show branding header
show_branding()

st.title("Welcome Page")
st.markdown("✅ You are logged in. Use the sidebar to navigate.")
