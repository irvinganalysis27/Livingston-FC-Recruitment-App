# ============================================================
# 🧠 SESSION MANAGER (persistent auth + page state)
# ============================================================
import streamlit as st
from datetime import datetime, timedelta

SESSION_TIMEOUT_HOURS = 24  # keep login active for 24h

def init_session():
    """Ensure all core session keys exist."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "login_time" not in st.session_state:
        st.session_state["login_time"] = None

def is_session_valid():
    """Return True if user is logged in and session not expired."""
    init_session()

    if not st.session_state["authenticated"]:
        return False

    login_time = st.session_state.get("login_time")
    if not login_time:
        return False

    elapsed = datetime.now() - login_time
    if elapsed > timedelta(hours=SESSION_TIMEOUT_HOURS):
        # expired
        reset_session()
        return False

    return True

def start_session():
    """Start a new authenticated session."""
    st.session_state["authenticated"] = True
    st.session_state["login_time"] = datetime.now()

def reset_session():
    """Force logout and clear stored state."""
    st.session_state.clear()
    st.session_state["authenticated"] = False
    st.session_state["login_time"] = None

def save_ui_state(**kwargs):
    """Save UI elements (filters, toggles, etc.) persistently across pages."""
    for k, v in kwargs.items():
        st.session_state[k] = v

def get_ui_state(key, default=None):
    """Retrieve saved UI element state."""
    return st.session_state.get(key, default)
