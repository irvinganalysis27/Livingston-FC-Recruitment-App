import streamlit as st

def check_password():
    if "password_ok" not in st.session_state:
        st.session_state["password_ok"] = False

    if not st.session_state["password_ok"]:
        st.markdown("## Welcome to the Livingston FC Recruitment App")
        password = st.text_input("Enter password:", type="password")
        if st.button("Login"):
            if password == "Livi2025":
                st.session_state["password_ok"] = True
                st.rerun()
            else:
                st.warning("Please enter the correct password to access the app.")
        st.stop()
