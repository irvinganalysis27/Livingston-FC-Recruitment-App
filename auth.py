import streamlit as st
from datetime import datetime, timedelta
from streamlit_js_eval import streamlit_js_eval
from lib.session_manager import start_session, is_session_valid, reset_session

PASSWORD = "Livi2025"
SESSION_HOURS = 24


def check_password():
    """
    Password protection that persists across all pages and page reloads for 24 hours.
    Works per-browser using localStorage.
    """
    # ============================================================
    # 🧠 1️⃣ Check local storage (browser side)
    # ============================================================
    stored_data = streamlit_js_eval(js_expressions="localStorage.getItem('livi_auth')", key="load_auth")

    if stored_data:
        try:
            token, timestamp = stored_data.split("|")
            if token == PASSWORD:
                login_time = datetime.fromisoformat(timestamp)
                if datetime.now() - login_time < timedelta(hours=SESSION_HOURS):
                    if not st.session_state.get("authenticated", False):
                        start_session()
                    return True
                else:
                    # expired, clear local storage
                    streamlit_js_eval(js_expressions="localStorage.removeItem('livi_auth')", key="clear_expired")
        except Exception:
            pass

    # ============================================================
    # 🧠 2️⃣ If already logged in within session memory
    # ============================================================
    if is_session_valid():
        return True

    # ============================================================
    # 🔒 3️⃣ Ask for password if not yet authenticated
    # ============================================================
    st.markdown("## Welcome to the Livingston FC Recruitment App")
    password = st.text_input("Enter password:", type="password")

    if password == PASSWORD:
        # start Python session
        start_session()
        # store token in browser localStorage (persist 24h)
        now = datetime.now().isoformat()
        streamlit_js_eval(
            js_expressions=f"localStorage.setItem('livi_auth', '{PASSWORD}|{now}')",
            key="save_auth"
        )
        st.success("✅ Access granted — you’ll stay logged in for 24 hours.")
        st.rerun()
        return True
    elif password:
        st.warning("❌ Incorrect password")

    return False
