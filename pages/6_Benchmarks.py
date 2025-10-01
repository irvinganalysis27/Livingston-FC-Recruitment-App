import streamlit as st
from auth import check_password

check_password()

st.title("Benchmarks Page")
st.write("This page will show benchmarks such as average, 75th percentile, and 90th percentile for each position.")
