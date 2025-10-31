# ============================================================
# 🧠 SESSION MANAGER (persistent auth + page state)
# ============================================================
import streamlit as st
from datetime import datetime, timedelta

SESSION_TIMEOUT_HOURS = 24  # keep login active for 24h


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
        reset_session(safe=True)
        return False

    return True


def start_session():
    """Start a new authenticated session."""
    init_session()
    st.session_state["authenticated"] = True
    st.session_state["login_time"] = datetime.now()
    st.session_state["auth_timestamp"] = datetime.now()


def reset_session(safe=False):
    """Force logout safely without breaking the Streamlit state."""
    if safe:
        for k in ["authenticated", "login_time", "auth_timestamp"]:
            st.session_state[k] = None
        st.session_state["authenticated"] = False
    else:
        st.session_state.clear()
        st.session_state["authenticated"] = False
        st.session_state["login_time"] = None
        st.session_state["auth_timestamp"] = None


def save_ui_state(**kwargs):
    """Save UI elements (filters, toggles, etc.) persistently across pages."""
    for k, v in kwargs.items():
        st.session_state[k] = v


def get_ui_state(key, default=None):
    """Retrieve saved UI element state."""
    return st.session_state.get(key, default)
