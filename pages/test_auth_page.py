import streamlit as st
from auth import check_password

st.title("Auth Debug Test")

try:
    ok = check_password()
    st.write("check_password() returned:", ok)
except Exception as e:
    st.error(f"❌ Crash in check_password: {e}")
