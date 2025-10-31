# auth.py
import streamlit as st
from datetime import datetime, timedelta

PASSWORD = "Livi2025"
SESSION_HOURS = 24


def check_password():
    """
    Password protection that persists across pages for up to 24 hours
    (even after reruns or navigation, as long as the Streamlit server stays alive).
    """

    # 1️⃣ If we already have a stored login timestamp
    if "auth_timestamp" in st.session_state:
        ts = st.session_state["auth_timestamp"]
        if datetime.now() - ts < timedelta(hours=SESSION_HOURS):
            st.session_state["authenticated"] = True
            return True
        else:
            # expired
            st.session_state["authenticated"] = False
            del st.session_state["auth_timestamp"]

    # 2️⃣ If session still valid from current runtime
    if st.session_state.get("authenticated", False):
        return True

    # 3️⃣ Otherwise prompt for password
    st.markdown("## Welcome to the Livingston FC Recruitment App")
    password = st.text_input("Enter password:", type="password")

    if password == PASSWORD:
        st.session_state["authenticated"] = True
        st.session_state["auth_timestamp"] = datetime.now()
        st.success("✅ Access granted — you’ll stay logged in for 24 hours.")
        st.rerun()
        return True
    elif password:
        st.warning("❌ Incorrect password")

    return False


def logout_button(label: str = "Logout"):
    """Manual logout (clears session immediately)."""
    if st.button(label, type="secondary"):
        st.session_state["authenticated"] = False
        if "auth_timestamp" in st.session_state:
            del st.session_state["auth_timestamp"]
        st.success("Logged out.")
        st.rerun()
