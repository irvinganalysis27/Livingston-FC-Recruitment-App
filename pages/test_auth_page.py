import streamlit as st
from datetime import datetime
st.title("Debug auth import")

try:
    from auth import check_password
    st.success("✅ Imported auth.py successfully")
except Exception as e:
    st.error(f"❌ Crash while importing auth: {e}")
    st.stop()

try:
    ok = check_password()
    st.write("check_password() executed:", ok)
except Exception as e:
    st.error(f"❌ Crash inside check_password(): {e}")
