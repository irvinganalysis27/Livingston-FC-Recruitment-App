# lib/session_manager.py
import streamlit as st
from datetime import datetime, timedelta

SESSION_TIMEOUT_HOURS = 24

def init_session():
    """Ensure all core session keys exist."""
    for key, default in {
        "authenticated": False,
        "login_time": None,
        "auth_timestamp": None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

def is_session_valid():
    """Return True if user is logged in and session not expired."""
    init_session()
    if not st.session_state.get("authenticated", False):
        return False

    ts = st.session_state.get("login_time") or st.session_state.get("auth_timestamp")
    if not ts:
        return False

    if isinstance(ts, datetime) and (datetime.now() - ts > timedelta(hours=SESSION_TIMEOUT_HOURS)):
        st.session_state["authenticated"] = False
        st.session_state["auth_timestamp"] = None
        st.session_state["login_time"] = None
        return False

    return True

def start_session():
    """Start a new authenticated session."""
    init_session()
    st.session_state["authenticated"] = True
    now = datetime.now()
    st.session_state["login_time"] = now
    st.session_state["auth_timestamp"] = now
