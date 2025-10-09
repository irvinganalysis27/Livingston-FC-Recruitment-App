import streamlit as st
from auth import check_password

# ---------- Run password check first ----------
if not check_password():
    st.stop()

# ---------- Sidebar UI (appears after login) ----------
with st.sidebar:
    st.header("⚙️ App Controls")
    if st.button("🔁 Force Reload Data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("✅ Cache cleared. Reloading app...")
        st.rerun()

# ---------- Main Page ----------
st.title("Welcome Page")
st.markdown("✅ You are logged in. Use the sidebar to navigate between pages.")
