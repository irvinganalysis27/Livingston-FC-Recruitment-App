import streamlit as st
from auth import check_password
from branding import show_branding

# Protect page
from auth import check_password
if not check_password():
    st.stop()

# Show branding header
show_branding()

st.title("Weightings Page")
st.write("This page will have sliders to adjust metric weightings by position and recalculate rankings live.")
