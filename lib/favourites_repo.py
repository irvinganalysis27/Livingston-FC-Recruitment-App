from datetime import datetime
import streamlit as st
from supabase import create_client

@st.cache_resource(show_spinner=False)
def get_supabase():
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["service_key"]
        return create_client(url, key)
    except Exception as e:
        st.error(f"Supabase init failed: {e}")
        return None

sb = get_supabase()
TABLE = "favourites"

def upsert_favourite(player, team, league, position, colour="", comment="", visible=True, updated_by=None, source="app"):
    if not sb:
        return False
    payload = {
        "player": player,
        "team": team,
        "league": league,
        "position": position,
        "colour": colour,
        "comment": comment,
        "visible": bool(visible),
        "updated_at": datetime.utcnow().isoformat(),
        "updated_by": updated_by or "",
        "source": source,
    }
    try:
        sb.table(TABLE).upsert(payload, on_conflict="player").execute()
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save favourite {player}: {e}")
        return False

def list_favourites(only_visible=True):
    if not sb:
        return []
    q = sb.table(TABLE).select("*").order("player", desc=False)
    if only_visible:
        q = q.eq("visible", True)
    try:
        return q.execute().data or []
    except Exception as e:
        print(f"[ERROR] list_favourites failed: {e}")
        return []

def delete_favourite(player):
    if not sb:
        return False
    try:
        sb.table(TABLE).delete().eq("player", player).execute()
        return True
    except Exception as e:
        print(f"[ERROR] delete_favourite failed: {e}")
        return False
