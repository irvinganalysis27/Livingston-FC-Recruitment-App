import streamlit as st
from auth import check_password
from branding import show_branding

# Protect page
from auth import check_password
if not check_password():
    st.stop()

# Show branding header
show_branding()

st.title("Team Rankings Page")
st.write("This page will show a team’s best XI by ranking and allow comparison of league-only vs whole dataset.")
