import streamlit as st

st.set_page_config(page_title="Livingston FC Recruitment App", layout="wide")

def check_password():
    """Return True if the user entered the correct password, False otherwise."""
    if "password_ok" not in st.session_state:
        st.session_state["password_ok"] = False

    if not st.session_state["password_ok"]:
        st.markdown("## Welcome to the Livingston FC Recruitment App")
        password = st.text_input("Enter password:", type="password")
        if st.button("Login"):
            if password == "Livi2025":  # <--- change here if you want a new password
                st.session_state["password_ok"] = True
                st.rerun()
            else:
                st.warning("Please enter the correct password to access the app.")
        st.stop()

    # ✅ Hide "app.py" sidebar entry once logged in
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] ul li:first-child {display: none;}
        </style>
        """,
        unsafe_allow_html=True
    )

    return True
