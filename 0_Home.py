import streamlit as st
from auth import check_password
from branding import show_branding

# ---------- Page setup ----------
st.set_page_config(page_title="Livingston FC Recruitment App", layout="wide")

# ---------- Password protection ----------
if not check_password():
    st.stop()

# ---------- Branding header ----------
show_branding()

# ---------- Main content ----------
st.title("Welcome Page")
st.markdown("✅ You are logged in. Use the sidebar to navigate between pages.")
