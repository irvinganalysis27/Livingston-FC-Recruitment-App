import streamlit as st
from auth import check_password
from branding import show_branding

# Protect page
from auth import check_password
if not check_password():
    st.stop()

# Show branding header
show_branding()

st.title("Best Players by Position")
st.write("This page will show leaderboards of the best players per position with filters for leagues and minutes played.")
