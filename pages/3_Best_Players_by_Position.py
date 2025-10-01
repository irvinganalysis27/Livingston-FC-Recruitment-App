import streamlit as st
from auth import check_password

if not check_password():
    st.stop()

st.title("Best Players by Position Page")
st.write("This page will show leaderboards of the best players per position with filters for leagues and minutes played.")
