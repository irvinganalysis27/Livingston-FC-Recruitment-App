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
    Fully safe for Streamlit Cloud – avoids JS/rerun race conditions.
    """

    # 1️⃣ Handle post-login localStorage write (delayed)
    if st.session_state.get("pending_login", False):
        try:
            now_iso = datetime.now().isoformat()
            streamlit_js_eval(
                js_expressions=f"localStorage.setItem('{LOCAL_KEY}', '{PASSWORD}|{now_iso}')",
                key=f"auth_save_{int(time.time())}"
            )
            st.session_state["pending_login"] = False
            st.session_state["authenticated"] = True
            st.success("✅ Access granted — you’ll stay logged in for 24 hours.")
        except Exception:
            st.warning("Login succeeded, but token couldn’t be stored.")
        return True

    # 2️⃣ Restore from localStorage if available
    try:
        stored = streamlit_js_eval(
            js_expressions=f"localStorage.getItem('{LOCAL_KEY}')",
            key="auth_load_local",
            want_output=True,
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
                        key="auth_clear_expired",
                    )
        except Exception:
            pass

    # 3️⃣ Already authenticated in this Streamlit session
    if is_session_valid():
        return True

    # 4️⃣ Ask for password
    st.markdown("## Welcome to the Livingston FC Recruitment App")
    pwd = st.text_input("Enter password:", type="password")

    if pwd == PASSWORD:
        start_session()
        st.session_state["pending_login"] = True  # trigger write next render
        st.experimental_rerun()  # safe rerun (no JS yet)
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
                key=f"auth_logout_{int(time.time())}",
            )
        except Exception:
            pass
        reset_session()
        st.success("Logged out.")
        time.sleep(0.3)
        st.rerun()
