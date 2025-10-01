import streamlit as st
from auth import check_password

check_password()

st.title("Team Rankings Page")
st.write("This page will show a team’s best XI by ranking and allow comparison of league-only vs whole dataset.")
