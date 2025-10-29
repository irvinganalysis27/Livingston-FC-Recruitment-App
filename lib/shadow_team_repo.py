from supabase import create_client
import streamlit as st

client = create_client(st.secrets["supabase"]["url"], st.secrets["supabase"]["key"])

def list_shadow_team():
    res = client.table("shadow_team").select("*").execute()
    return res.data or []

def upsert_shadow_team(payload):
    try:
        client.table("shadow_team").upsert(payload, on_conflict=["player"]).execute()
        return True
    except Exception as e:
        print("Error upserting shadow team:", e)
        return False

def delete_shadow_team(player):
    try:
        client.table("shadow_team").delete().eq("player", player).execute()
        return True
    except Exception as e:
        print("Error deleting shadow team:", e)
        return False
