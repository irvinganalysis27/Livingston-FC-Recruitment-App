# auth.py

import streamlit as st
from datetime import datetime, timedelta
from lib.session_manager import start_session, is_session_valid, reset_session

PASSWORD = "Livi2025"
SESSION_HOURS = 24


def check_password():
    """
    Password authentication with 24-hour persistence within a browser session.
    Works across all pages and survives reloads as long as tab stays open.
    """

    # 1️⃣ If already authenticated and still valid
    if "auth_timestamp" in st.session_state:
        if datetime.now() - st.session_state["auth_timestamp"] < timedelta(hours=SESSION_HOURS):
            st.session_state["authenticated"] = True
            return True
        else:
            reset_session()
            st.session_state.clear()

    # 2️⃣ If user is already authenticated in this session
    if is_session_valid():
        return True

    # 3️⃣ Prompt for password
    st.markdown("## Welcome to the Livingston FC Recruitment App")
    password = st.text_input("Enter password:", type="password")

    if password == PASSWORD:
        start_session()
        st.session_state["auth_timestamp"] = datetime.now()
        st.success("✅ Access granted — you’ll stay logged in for 24 hours (or until you close the tab).")
        st.rerun()
        return True
    elif password:
        st.warning("❌ Incorrect password")

    return False


def logout_button(label: str = "Logout"):
    """Manual logout (clears session immediately)."""
    if st.button(label, type="secondary"):
        reset_session()
        st.session_state.clear()
        st.success("Logged out.")
        st.rerun()
