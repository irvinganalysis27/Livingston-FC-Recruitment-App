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

    # ✅ Safe initialise keys
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "auth_timestamp" not in st.session_state:
        st.session_state["auth_timestamp"] = None

    # 1️⃣ Check if still valid
    ts = st.session_state["auth_timestamp"]
    if ts and isinstance(ts, datetime):
        if datetime.now() - ts < timedelta(hours=SESSION_HOURS):
            st.session_state["authenticated"] = True
            return True
        else:
            # expired session — don’t clear everything mid-run, just flag it
            st.session_state["authenticated"] = False
            st.session_state["auth_timestamp"] = None

    # 2️⃣ If session still valid
    if is_session_valid():
        return True

    # 3️⃣ Show login box
    st.markdown("## Welcome to the Livingston FC Recruitment App")
    password = st.text_input("Enter password:", type="password")

    if password == PASSWORD:
        start_session()
        st.session_state["auth_timestamp"] = datetime.now()
        st.success("✅ Access granted — you’ll stay logged in for 24 hours (or until tab closes).")
        st.rerun()
        return True
    elif password:
        st.warning("❌ Incorrect password")

    return False


def logout_button(label: str = "Logout"):
    """Manual logout (clears session safely)."""
    if st.button(label, type="secondary"):
        reset_session()
        st.session_state["authenticated"] = False
        st.session_state["auth_timestamp"] = None
        st.success("Logged out.")
        st.rerun()
