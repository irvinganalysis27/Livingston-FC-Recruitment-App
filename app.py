import streamlit as st
from auth import check_password

st.set_page_config(page_title="Livingston FC Recruitment App", layout="wide")

# Run auth check at the very start
if not check_password():
    st.stop()

# If login OK, send user straight to Home
st.switch_page("pages/0_Home.py")
