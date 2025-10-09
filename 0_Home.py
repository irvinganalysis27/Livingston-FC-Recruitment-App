import streamlit as st
from auth import check_password
from branding import show_branding

st.set_page_config(page_title="Livingston FC Recruitment App", layout="wide")

# ---------- Protect page ----------
if not check_password():
    st.stop()

# ---------- Show branding header ----------
show_branding()

# ---------- Sidebar Reload Button ----------
with st.sidebar:
    st.header("⚙️ App Controls")
    if st.button("🔁 Force Reload Data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("✅ Cache cleared, reloading app...")
        st.rerun()

# ---------- Main Page ----------
st.title("Welcome Page")
st.markdown("✅ You are logged in. Use the sidebar to navigate.")
