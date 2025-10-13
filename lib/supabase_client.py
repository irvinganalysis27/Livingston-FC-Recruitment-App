import streamlit as st
from supabase import create_client, Client

@st.cache_resource(show_spinner=False)
def get_supabase() -> Client:
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"].get("service_key")  # server-side key
    return create_client(url, key)
