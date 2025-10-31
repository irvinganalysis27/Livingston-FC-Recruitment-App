# ============================================================
# 🧠 SESSION MANAGER — page state persistence (filters, toggles, etc.)
# ============================================================

import streamlit as st

def save_ui_state(**kwargs):
    """
    Save UI values so they persist across pages.
    Example: save_ui_state(status_filter=selected_statuses)
    """
    for key, value in kwargs.items():
        st.session_state[key] = value

def get_ui_state(key, default=None):
    """
    Retrieve saved UI values safely.
    Example: selected_statuses = get_ui_state('status_filter', ['🟣 Needs Checked'])
    """
    return st.session_state.get(key, default)

def reset_ui_state(*keys):
    """
    Reset one or more UI values.
    Example: reset_ui_state('status_filter', 'fav_pos_groups')
    """
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]
