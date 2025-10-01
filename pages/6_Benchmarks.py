import streamlit as st
from auth import check_password
from branding import show_branding

# Protect page
from auth import check_password
if not check_password():
    st.stop()

# Show branding header
show_branding()

st.title("Benchmarks Page")
st.write("This page will show benchmarks such as average, 75th percentile, and 90th percentile for each position.")
