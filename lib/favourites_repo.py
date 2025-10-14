from datetime import datetime
import time
import streamlit as st
from supabase import create_client
import gspread
from google.oauth2.service_account import Credentials

TABLE = "favourites"


# ============================================================
# 🔗 Supabase Connection (fresh client per call)
# ============================================================

def get_supabase_client():
    """Always return a new Supabase client to avoid thread blocking."""
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["service_key"]
        return create_client(url, key)
    except Exception as e:
        st.error(f"❌ Failed to connect to Supabase: {e}")
        return None


# ============================================================
# 🛡️ Safe Execute Helper
# ============================================================

def safe_execute(query, retries=3, delay=0.3):
    """Retry a Supabase query a few times if connection is busy."""
    for i in range(retries):
        try:
            return query.execute()
        except Exception as e:
            if "Resource temporarily unavailable" in str(e) and i < retries - 1:
                time.sleep(delay)
                continue
            raise


# ============================================================
# 🧾 Google Sheet Logging (optional, non-blocking placeholder)
# ============================================================

def append_to_google_sheet(record):
    """Append a favourite change to Google Sheet log (optional)."""
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

        # Currently disabled for speed/debugging
        # sheet.append_row(row, value_input_option="USER_ENTERED")
        print(f"[LOG] (simulated) Added record for {record.get('player')}")
    except Exception as e:
        print(f"[ERROR] Google Sheet log failed: {e}")


# ============================================================
# 💾 Supabase CRUD Functions
# ============================================================

def upsert_favourite(record):
    sb = get_supabase_client()
    if not sb:
        print("[ERROR] Supabase client not available.")
        return False

    payload = {
        "player": record.get("player"),
        "team": record.get("team"),
        "league": record.get("league"),
        "position": record.get("position"),
        "colour": record.get("colour", "🟣 Needs Checked"),
        "comment": record.get("comment", ""),
        "visible": bool(record.get("visible", True)),
        "updated_at": datetime.utcnow().isoformat(),
        "updated_by": record.get("updated_by", "auto"),
        "source": record.get("source", "radar-page"),
    }

    try:
        safe_execute(sb.table(TABLE).upsert(payload, on_conflict="player"))
        append_to_google_sheet(payload)
        print(f"[INFO] ✅ Upserted favourite for {payload['player']}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to upsert favourite {payload['player']}: {e}")
        return False


def list_favourites(only_visible=True):
    sb = get_supabase_client()
    if not sb:
        return []

    q = sb.table(TABLE).select("*").order("player", desc=False)
    if only_visible:
        q = q.eq("visible", True)

    try:
        res = safe_execute(q)
        return res.data or []
    except Exception as e:
        print(f"[ERROR] list_favourites failed: {e}")
        return []


def delete_favourite(player):
    sb = get_supabase_client()
    if not sb:
        return False
    try:
        safe_execute(sb.table(TABLE).delete().eq("player", player))
        print(f"[INFO] Deleted favourite: {player}")
        return True
    except Exception as e:
        print(f"[ERROR] delete_favourite failed for {player}: {e}")
        return False


def hide_favourite(player):
    sb = get_supabase_client()
    if not sb:
        return False
    try:
        safe_execute(
            sb.table(TABLE)
            .update({"visible": False, "updated_at": datetime.utcnow().isoformat()})
            .eq("player", player)
        )
        print(f"[INFO] Hid favourite: {player}")
        return True
    except Exception as e:
        print(f"[ERROR] hide_favourite failed for {player}: {e}")
        return False
