from datetime import datetime, timezone
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
# 🧾 Google Sheet Logging (auto-create + safe append)
# ============================================================

@st.cache_resource(show_spinner=False)
def get_gsheet():
    """Authorize and return the Google Sheet worksheet handle."""
    try:
        creds_info = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(
            creds_info,
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        gc = gspread.authorize(creds)

        spreadsheet_id = "1ESiZsk7W-LrotYs7hpznJB4K-dHgKA0bWE1oUCJ8Pf0"

        try:
            sheet = gc.open_by_key(spreadsheet_id).worksheet("favourites_log")
        except gspread.WorksheetNotFound:
            sh = gc.open_by_key(spreadsheet_id)
            sheet = sh.add_worksheet(title="favourites_log", rows=1000, cols=11)
            sheet.append_row([
                "Timestamp", "Player", "Team", "League", "Position",
                "Colour", "Initial Watch", "Second Watch", "Latest Action",
                "Visible", "Updated_by", "Source"
            ], value_input_option="USER_ENTERED")
            print("[INFO] Created new 'favourites_log' worksheet")

        return sheet

    except Exception as e:
        print(f"[ERROR] Could not connect to Google Sheets: {e}")
        return None


def append_to_google_sheet(record):
    """Append a favourite change to Google Sheet log."""
    sheet = get_gsheet()
    if not sheet:
        print("[WARN] No sheet handle, skipping log.")
        return

    try:
        row = [
            datetime.now(timezone.utc).isoformat(),
            record.get("player", ""),
            record.get("team", ""),
            record.get("league", ""),
            record.get("position", ""),
            record.get("colour", ""),
            record.get("initial_watch_comment", ""),
            record.get("second_watch_comment", ""),
            record.get("latest_action", ""),
            record.get("visible", True),
            record.get("updated_by", "auto"),
            record.get("source", "radar-page"),
        ]
        sheet.append_row(row, value_input_option="USER_ENTERED")
        print(f"[LOG] ✅ Logged to Google Sheet for {record.get('player')}")
    except Exception as e:
        print(f"[ERROR] Google Sheet log failed: {e}")


# ============================================================
# 💾 Supabase CRUD Functions
# ============================================================

def upsert_favourite(record, log_to_sheet=False):
    """
    Insert or update a favourite in Supabase.
    Only performs an update if something has actually changed.
    Logs to Google Sheets if log_to_sheet=True.
    """
    sb = get_supabase_client()
    if not sb:
        print("[ERROR] Supabase client not available.")
        return False

    player = record.get("player")
    if not player:
        print("[WARN] No player provided in record.")
        return False

    payload = {
        "player": player,
        "team": record.get("team"),
        "league": record.get("league"),
        "position": record.get("position"),
        "colour": record.get("colour", "🟣 Needs Checked"),
        "initial_watch_comment": record.get("initial_watch_comment", ""),
        "second_watch_comment": record.get("second_watch_comment", ""),
        "latest_action": record.get("latest_action", ""),
        "visible": bool(record.get("visible", True)),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "updated_by": record.get("updated_by", "auto"),
        "source": record.get("source", "radar-page"),
    }

    try:
        # --- 1️⃣ Fetch existing record ---
        existing = sb.table(TABLE).select("*").eq("player", player).execute()
        existing_row = existing.data[0] if existing.data else None

        # --- 2️⃣ Skip identical data ---
        if existing_row:
            changed = any(
                str(existing_row.get(k, "")).strip() != str(payload.get(k, "")).strip()
                for k in [
                    "team", "league", "position", "colour",
                    "initial_watch_comment", "second_watch_comment", "visible"
                ]
            )
            if not changed:
                print(f"[DEBUG] Skipping {player} — no change detected")
                return True

        # --- 3️⃣ Perform upsert if new/changed ---
        safe_execute(sb.table(TABLE).upsert(payload, on_conflict="player"))
        print(f"[INFO] ✅ Upserted favourite for {player}")

        if log_to_sheet:
            append_to_google_sheet(payload)

        return True

    except Exception as e:
        print(f"[ERROR] Failed to upsert favourite {player}: {e}")
        return False


def list_favourites(only_visible=True):
    """Return all favourites (optionally only visible ones)."""
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
    """Permanently delete a favourite."""
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
    """Hide a favourite without deleting it."""
    sb = get_supabase_client()
    if not sb:
        return False
    try:
        safe_execute(
            sb.table(TABLE)
            .update({
                "visible": False,
                "updated_at": datetime.now(timezone.utc).isoformat()
            })
            .eq("player", player)
        )
        print(f"[INFO] Hid favourite: {player}")
        return True
    except Exception as e:
        print(f"[ERROR] hide_favourite failed for {player}: {e}")
        return False
