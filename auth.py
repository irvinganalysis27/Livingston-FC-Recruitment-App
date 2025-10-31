# auth.py
import streamlit as st
from datetime import datetime, timedelta
from lib.session_manager import start_session, is_session_valid

PASSWORD = "Livi2025"
SESSION_HOURS = 24


def check_password():
    """
    Password authentication with 24-hour persistence per tab.
    Safe on reload and across pages — no crashes.
    """

    # Initialise keys safely
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "auth_timestamp" not in st.session_state:
        st.session_state["auth_timestamp"] = None

    # ✅ If already authenticated and still valid
    ts = st.session_state["auth_timestamp"]
    if st.session_state.get("authenticated", False) and isinstance(ts, datetime):
        if datetime.now() - ts < timedelta(hours=SESSION_HOURS):
            return True
        else:
            # session expired – mark invalid (don’t clear everything)
            st.session_state["authenticated"] = False
            st.session_state["auth_timestamp"] = None

    # ✅ If session_manager already says valid
    if is_session_valid():
        return True

    # 🔒 Password prompt
    st.markdown("## Welcome to the Livingston FC Recruitment App")
    password = st.text_input("Enter password:", type="password")

    # Only process input after user types something
    if password:
        if password == PASSWORD:
            start_session()
            st.session_state["auth_timestamp"] = datetime.now()
            st.session_state["authenticated"] = True
            st.success("✅ Access granted — valid for 24 hours or until you close the tab.")
            st.experimental_set_query_params(auth="1")  # prevent reload loop
            st.rerun()
            return True
        else:
            st.warning("❌ Incorrect password")

    return False


def logout_button(label: str = "Logout"):
    """Manual logout that doesn't break session."""
    if st.button(label, type="secondary"):
        st.session_state["authenticated"] = False
        st.session_state["auth_timestamp"] = None
        st.experimental_set_query_params(clear="1")
        st.success("Logged out.")
        st.rerun()
