import streamlit as st
from lib.session_manager import is_session_valid, start_session, reset_session

PASSWORD = "Livi2025"

def check_password():
    """
    Password protection that persists across all pages for 24 hours.
    """
    # --- Already authenticated and session valid ---
    if is_session_valid():
        return True

    # --- If expired or not logged in ---
    st.markdown("## Welcome to the Livingston FC Recruitment App")
    password = st.text_input("Enter password:", type="password")

    if password == PASSWORD:
        start_session()
        st.success("✅ Access granted — valid for 24 hours.")
        st.rerun()  # refresh after login
        return True
    elif password:
        st.warning("❌ Incorrect password")

    return False
