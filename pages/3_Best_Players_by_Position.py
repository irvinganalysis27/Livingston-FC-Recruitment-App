import streamlit as st
from auth import check_password

check_password()

st.title("Best Players by Position Page")
st.write("This page will show leaderboards of the best players per position with filters for leagues and minutes played.")
