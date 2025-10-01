import streamlit as st
from auth import check_password

# Set global page config
st.set_page_config(page_title="Livingston FC Recruitment App", layout="wide")

# Run password check
if not check_password():
    st.stop()

# If logged in, show welcome page (not clickable in sidebar)
st.title("Welcome to the Livingston FC Recruitment App")
st.markdown("✅ You are logged in. Use the sidebar to access the tools.")

# Optional: auto-redirect to 0_Home on login
# This avoids confusion by jumping them to Home immediately
if "just_logged_in" not in st.session_state:
    st.session_state.just_logged_in = True
    st.switch_page("pages/0_Home.py")
