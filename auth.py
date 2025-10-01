import streamlit as st

PASSWORD = "Livi2025"

def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        pwd = st.text_input("Enter password:", type="password")
        if pwd == PASSWORD:
            st.session_state.authenticated = True
            st.sidebar.success("Logged in!")
        else:
            st.stop()
    else:
        st.sidebar.success("Logged in!")
