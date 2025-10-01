import streamlit as st

PASSWORD = "Livi2025"
st.set_page_config(page_title="Livingston FC Recruitment App", layout="centered")

st.sidebar.title("Welcome Page")
st.title("Welcome to the Livingston FC Recruitment App")

pwd = st.text_input("Enter password:", type="password")
if pwd != PASSWORD:
    st.warning("Please enter the correct password to access the app.")
    st.stop()
else:
    st.session_state["authenticated"] = True
    st.success("✅ Logged in! Use the sidebar to navigate to different pages.")
