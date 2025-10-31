# auth.py
import streamlit as st
from datetime import datetime, timedelta
from lib.session_manager import start_session, is_session_valid

PASSWORD = "Livi2025"
SESSION_HOURS = 24


def check_password():
    """
    Password authentication that:
    - Persists across pages and reloads for 24 hours.
    - Never crashes or triggers double reruns.
    """

    # --- Safe initialisation ---
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "auth_timestamp" not in st.session_state:
        st.session_state["auth_timestamp"] = None
    if "auth_prompt_shown" not in st.session_state:
        st.session_state["auth_prompt_shown"] = False

    # --- Valid existing session ---
    ts = st.session_state["auth_timestamp"]
    if st.session_state["authenticated"] and ts and isinstance(ts, datetime):
        if datetime.now() - ts < timedelta(hours=SESSION_HOURS):
            return True
        else:
            st.session_state["authenticated"] = False
            st.session_state["auth_timestamp"] = None

    # --- Valid session via session_manager ---
    if is_session_valid():
        return True

    # --- Only show the password input once ---
    if not st.session_state["auth_prompt_shown"]:
        st.markdown("## Welcome to the Livingston FC Recruitment App")
        password = st.text_input("Enter password:", type="password")
        st.session_state["auth_prompt_shown"] = True
    else:
        # Retrieve previous value so Streamlit doesn't rerun endlessly
        password = st.session_state.get("last_password_entry", "")

    if password:
        st.session_state["last_password_entry"] = password
        if password == PASSWORD:
            start_session()
            st.session_state["auth_timestamp"] = datetime.now()
            st.session_state["authenticated"] = True
            st.success("✅ Access granted — valid for 24 hours.")
            st.session_state["auth_prompt_shown"] = False
            st.experimental_rerun()
            return True
        else:
            st.warning("❌ Incorrect password")

    return False


def logout_button(label: str = "Logout"):
    """Manual logout."""
    if st.button(label, type="secondary"):
        st.session_state.clear()
        st.success("Logged out.")
        st.experimental_rerun()
