# ============================================================
# lib/shadow_team_repo.py
# ============================================================

import streamlit as st
from supabase import create_client

# --- Create Supabase client (reads from your app secrets) ---
url = st.secrets["supabase"]["url"]
key = st.secrets["supabase"]["service_key"]
client = create_client(url, key)

# ---------- List players ----------
def list_shadow_team():
    try:
        res = client.table("shadow_team").select("*").execute()
        return res.data or []
    except Exception as e:
        print(f"[shadow_team_repo] list_shadow_team error: {e}")
        return []

# ---------- Add or update player ----------
def upsert_shadow_team(payload: dict) -> bool:
    """Insert or update a shadow team record (unique on player)."""
    try:
        client.table("shadow_team").upsert(payload, on_conflict=["player"]).execute()
        return True
    except Exception as e:
        print(f"[shadow_team_repo] upsert error: {e}")
        return False

# ---------- Remove player ----------
def delete_shadow_team(player: str) -> bool:
    try:
        client.table("shadow_team").delete().eq("player", player).execute()
        return True
    except Exception as e:
        print(f"[shadow_team_repo] delete error: {e}")
        return False
