# auth.py

import time
import streamlit as st
from datetime import datetime, timedelta
from streamlit_js_eval import streamlit_js_eval
from lib.session_manager import start_session, is_session_valid, reset_session

PASSWORD = "Livi2025"
SESSION_HOURS = 24
LOCAL_KEY = "livi_auth"


def check_password() -> bool:
    """
    Persistent password authentication across reloads and pages (24-hour expiry).
    Safe for Streamlit Cloud and avoids race conditions.
    """

    # Handle post-JS rerun cycle
    if st.session_state.get("pending_rerun"):
        st.session_state["pending_rerun"] = False
        st.rerun()

    # 1️⃣ Try restore from localStorage
    try:
        stored = streamlit_js_eval(
            js_expressions=f"localStorage.getItem('{LOCAL_KEY}')",
            key="auth_load_local",
            want_output=True
        )
    except Exception:
        stored = None

    if stored:
        try:
            token, ts = stored.split("|", 1)
            if token == PASSWORD:
                login_time = datetime.fromisoformat(ts)
                if datetime.now() - login_time < timedelta(hours=SESSION_HOURS):
                    if not st.session_state.get("authenticated", False):
                        start_session()
                    return True
                else:
                    streamlit_js_eval(
                        js_expressions=f"localStorage.removeItem('{LOCAL_KEY}')",
                        key="auth_clear_expired"
                    )
        except Exception:
            pass

    # 2️⃣ Already authenticated in this Streamlit run
    if is_session_valid():
        return True

    # 3️⃣ Prompt user
    st.markdown("## Welcome to the Livingston FC Recruitment App")
    pwd = st.text_input("Enter password:", type="password")

    if pwd == PASSWORD:
        start_session()

        try:
            now_iso = datetime.now().isoformat()
            streamlit_js_eval(
                js_expressions=f"localStorage.setItem('{LOCAL_KEY}', '{PASSWORD}|{now_iso}')",
                key=f"auth_save_{int(time.time())}"
            )
            st.session_state["pending_rerun"] = True  # trigger rerun next cycle
            st.success("✅ Access granted — you'll stay logged in for 24 hours.")
            return True
        except Exception as e:
            st.warning(f"Login succeeded, but token save failed: {e}")
            return True

    elif pwd:
        st.warning("❌ Incorrect password")

    return False


def logout_button(label: str = "Logout"):
    """Optional logout helper (clears Streamlit + browser)."""
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
        time.sleep(0.3)
        st.rerun()
