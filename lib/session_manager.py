import streamlit as st
from datetime import datetime, timedelta

SESSION_TIMEOUT_HOURS = 24  # not actively used here but can stay

def init_session():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "login_time" not in st.session_state:
        st.session_state["login_time"] = None

def is_session_valid():
    init_session()
    return st.session_state.get("authenticated", False)

def start_session():
    st.session_state["authenticated"] = True
    st.session_state["login_time"] = datetime.now()

def reset_session():
    st.session_state["authenticated"] = False
    st.session_state["login_time"] = None
