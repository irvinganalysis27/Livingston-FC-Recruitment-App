# auth.py

import time
import streamlit as st
from datetime import datetime, timedelta
from streamlit_js_eval import streamlit_js_eval
from lib.session_manager import start_session, is_session_valid, reset_session

PASSWORD = "Livi2025"
SESSION_HOURS = 24
LOCAL_KEY = "livi_auth"  # stored as "password|iso_timestamp"


def check_password() -> bool:
    """
    Password protection that persists across all pages and reloads for 24 hours.
    Persistence is per browser via localStorage. Session state is also set so
    page navigation inside the app does not reset filters.
    """

    # 1) Try to restore from localStorage (browser side)
    try:
        stored = streamlit_js_eval(
            js_expressions=f"localStorage.getItem('{LOCAL_KEY}')",
            key="auth_load_local"
        )
    except Exception:
        stored = None

    if stored:
        try:
            token, ts = stored.split("|", 1)
            if token == PASSWORD:
                login_time = datetime.fromisoformat(ts)
                if datetime.now() - login_time < timedelta(hours=SESSION_HOURS):
                    # Make sure Streamlit session is marked as authenticated
                    if not st.session_state.get("authenticated", False):
                        start_session()
                    return True
                else:
                    # Expired, clear local storage
                    streamlit_js_eval(
                        js_expressions=f"localStorage.removeItem('{LOCAL_KEY}')",
                        key="auth_clear_expired"
                    )
        except Exception:
            # Ignore parse errors and fall through to password prompt
            pass

    # 2) If session is already valid (same tab during this run)
    if is_session_valid():
        return True

    # 3) Ask for password
    st.markdown("## Welcome to the Livingston FC Recruitment App")
    pwd = st.text_input("Enter password:", type="password")

    if pwd == PASSWORD:
        # Mark this Streamlit session as authenticated
        start_session()

        # Save a browser token so reloads and navigation stay logged in
        try:
            now_iso = datetime.now().isoformat()
            streamlit_js_eval(
                js_expressions=f"localStorage.setItem('{LOCAL_KEY}', '{PASSWORD}|{now_iso}')",
                key=f"auth_save_{int(time.time())}"
            )
            st.success("Access granted.")
            # Small pause so JS finishes before rerun
            time.sleep(0.4)
            st.rerun()
        except Exception as e:
            st.warning(f"Login succeeded, but could not persist token: {e}")
        return True

    elif pwd:
        st.warning("Incorrect password")

    return False


def logout_button(label: str = "Logout"):
    """
    Optional helper. Call this in any page to show a logout button that clears
    both Streamlit session state and the browser token.
    """
    if st.button(label, type="secondary"):
        try:
            streamlit_js_eval(
                js_expressions=f"localStorage.removeItem('{LOCAL_KEY}')",
                key=f"auth_logout_{int(time.time())}"
            )
        except Exception:
            pass
        reset_session()
        st.success("Logged out.")
        time.sleep(0.2)
        st.rerun()
