import streamlit as st

PASSWORD = "Livi2025"

def check_password():
    """Return True if correct password entered, remember in session."""
    if "password_ok" not in st.session_state:
        st.session_state.password_ok = False

    if st.session_state.password_ok:
        return True

    st.markdown("## Welcome to the Livingston FC Recruitment App")
    password = st.text_input("Enter password:", type="password")

    if password == PASSWORD:
        st.session_state.password_ok = True
        st.success("Access granted.")
        st.rerun()
        return True
    elif password:
        st.warning("Incorrect password.")
        return False

    return False
