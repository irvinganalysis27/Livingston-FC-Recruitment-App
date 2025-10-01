import streamlit as st
from auth import check_password

# Run password check globally
if not check_password():
    st.stop()

# This file just acts as a landing page
st.title("Welcome Page")
st.markdown("✅ You are logged in. Use the sidebar to navigate.")
