import streamlit as st
from auth import check_password

# Run auth once at the very top
if not check_password():
    st.stop()

# Redirect user to the welcome page
st.switch_page("pages/0_Home.py")
