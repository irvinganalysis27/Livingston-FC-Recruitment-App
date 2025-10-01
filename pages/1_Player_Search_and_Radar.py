import streamlit as st
from auth import check_password

check_password()

st.title("Player Search & Radar Page")
st.write("This page will let you search for a player and generate radar plots based on their position’s metric group.")
