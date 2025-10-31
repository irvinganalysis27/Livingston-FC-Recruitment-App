import streamlit as st
from datetime import datetime, timedelta
from lib.session_manager import start_session, is_session_valid, reset_session

PASSWORD = "Livi2025"

def check_password():
    """
    Password protection that persists across all pages.
    Once logged in, user stays authenticated until the Streamlit session ends or app is closed.
    """
    # --- Already authenticated ---
    if st.session_state.get("authenticated", False):
        return True

    # --- If not authenticated yet ---
    st.markdown("## Welcome to the Livingston FC Recruitment App")
    password = st.text_input("Enter password:", type="password")

    if password == PASSWORD:
        st.session_state["authenticated"] = True
        start_session()
        st.success("✅ Access granted")
        st.rerun()
        return True
    elif password:
        st.warning("❌ Incorrect password")

    return False
