import streamlit as st
from auth import check_password

# ---------- FORCE CACHE RESET ----------
if st.sidebar.button("🔁 Force Reload Data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# Run password check globally
if not check_password():
    st.stop()

# This file just acts as a landing page
st.title("Welcome Page")
st.markdown("✅ You are logged in. Use the sidebar to navigate.")
