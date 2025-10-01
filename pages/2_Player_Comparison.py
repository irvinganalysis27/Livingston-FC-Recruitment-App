import streamlit as st
from auth import check_password
from branding import show_branding

# Protect page
from auth import check_password
if not check_password():
    st.stop()

# Show branding header
show_branding()

st.title("Player Comparison Page")
st.write("This page will allow selecting 2–3 players and show side-by-side radars and comparison tables.")
