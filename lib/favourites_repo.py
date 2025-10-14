from datetime import datetime
import streamlit as st
from supabase import create_client
import gspread
from google.oauth2.service_account import Credentials

# ============================================================
# 🔗 Supabase Connection
# ============================================================

@st.cache_resource(show_spinner=False)
def get_supabase_client():
    """Return a cached Supabase client instance."""
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["service_key"]
        return create_client(url, key)
    except Exception as e:
        st.error(f"❌ Failed to connect to Supabase: {e}")
        return None

sb = get_supabase_client()
TABLE = "favourites"

# ============================================================
# 🧾 Google Sheet Logging
# ============================================================

def append_to_google_sheet(record):
    """Append a new favourite or update to the Google Sheet log."""
    try:
        creds_info = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(
            creds_info,
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        gc = gspread.authorize(creds)
        sheet = gc.open("Livingston_Favourites_Log").worksheet("favourites_log")

        row = [
            datetime.utcnow().isoformat(),
            record.get("player", ""),
            record.get("team", ""),
            record.get("league", ""),
            record.get("position", ""),
            record.get("colour", ""),
            record.get("comment", ""),
            record.get("visible", True),
            record.get("updated_by", "auto"),
            record.get("source", "radar-page"),
        ]
        sheet.append_row(row, value_input_option="USER_ENTERED")
        print(f"[LOG] ✅ Added record for {record.get('player')}")
    except Exception as e:
        print(f"[ERROR] Failed to write to Google Sheet: {e}")

# ============================================================
# 💾 Supabase Favourites Functions
# ============================================================

def upsert_favourite(record):
    """Insert or update a favourite player record in Supabase."""
    if not sb:
        print("[ERROR] Supabase client not available.")
        return False

    payload = {
        "player": record.get("player"),
        "team": record.get("team"),
        "league": record.get("league"),
        "position": record.get("position"),
        "colour": record.get("colour", ""),
        "comment": record.get("comment", ""),
        "visible": bool(record.get("visible", True)),
        "updated_at": datetime.utcnow().isoformat(),
        "updated_by": record.get("updated_by", "auto"),
        "source": record.get("source", "radar-page"),
    }

    try:
        sb.table(TABLE).upsert(payload, on_conflict="player").execute()
        append_to_google_sheet(payload)
        print(f"[INFO] ✅ Upserted favourite for {payload['player']}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to upsert favourite {payload['player']}: {e}")
        return False

def list_favourites(only_visible=True):
    """List all favourites from Supabase."""
    if not sb:
        print("[ERROR] Supabase client not available.")
        return []
    q = sb.table(TABLE).select("*").order("player", desc=False)
    if only_visible:
        q = q.eq("visible", True)
    try:
        res = q.execute()
        return res.data or []
    except Exception as e:
        print(f"[ERROR] list_favourites failed: {e}")
        return []

def delete_favourite(player):
    """Permanently delete a favourite player."""
    if not sb:
        print("[ERROR] Supabase client not available.")
        return False
    try:
        sb.table(TABLE).delete().eq("player", player).execute()
        print(f"[INFO] Deleted favourite: {player}")
        return True
    except Exception as e:
        print(f"[ERROR] delete_favourite failed for {player}: {e}")
        return False

def hide_favourite(player):
    """Mark a favourite as hidden (visible=False) instead of deleting."""
    if not sb:
        print("[ERROR] Supabase client not available.")
        return False
    try:
        sb.table(TABLE).update({
            "visible": False,
            "updated_at": datetime.utcnow().isoformat()
        }).eq("player", player).execute()
        print(f"[INFO] Hid favourite: {player}")
        return True
    except Exception as e:
        print(f"[ERROR] hide_favourite failed for {player}: {e}")
        return False
