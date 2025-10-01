import streamlit as st
from auth import check_password

# Set config first
st.set_page_config(page_title="Livingston FC Recruitment App", layout="wide")

# Run password check first
check_password()

# If logged in, just show a welcome message (acts like a landing page)
st.title("Welcome Page")
st.markdown("✅ You are logged in. Use the sidebar to navigate.")
