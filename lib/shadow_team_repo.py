# lib/shadow_team_repo.py
from datetime import datetime, timezone
import time
import streamlit as st
from supabase import create_client

TABLE = "shadow_team"

# ---------- Client (use service_key like favourites) ----------
def get_supabase_client():
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["service_key"]   # <- service_key, not key/anon
        return create_client(url, key)
    except Exception as e:
        st.error(f"❌ Supabase connection failed: {e}")
        return None

def safe_execute(query, retries=3, delay=0.3):
    for i in range(retries):
        try:
            return query.execute()
        except Exception as e:
            if "Resource temporarily unavailable" in str(e) and i < retries - 1:
                time.sleep(delay)
                continue
            # Bubble up exact error (so our page can print it)
            raise

# ---------- CRUD ----------
def upsert_shadow_team(record: dict) -> bool:
    """
    record = {"player": str, "position_slot": str, "rank": int}
    Upsert by player (one slot per player). Change on_conflict if you prefer another rule.
    """
    sb = get_supabase_client()
    if not sb:
        return False

    player = (record.get("player") or "").strip()
    if not player:
        print("[shadow_team_repo] No 'player' provided.")
        return False

    payload = {
        "player": player,
        "position_slot": record.get("position_slot") or "ST",
        "rank": int(record.get("rank") or 0),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        # Optional: Fetch existing to avoid no-op updates
        existing = sb.table(TABLE).select("*").eq("player", player).execute()
        if existing.data:
            # Update
            safe_execute(
                sb.table(TABLE)
                .update(payload)
                .eq("player", player)
            )
        else:
            # Insert (use on_conflict if you have a unique index on player)
            safe_execute(
                sb.table(TABLE).upsert(payload, on_conflict="player")
            )
        print(f"[shadow_team_repo] ✅ Upsert OK for {player} -> {payload}")
        return True

    except Exception as e:
        # Show full error in logs and a small hint in the UI
        print(f"[shadow_team_repo] ❌ Upsert failed for {player}: {e}")
        st.error(f"Shadow Team upsert failed for {player}. Check logs for details.")
        return False


def list_shadow_team():
    sb = get_supabase_client()
    if not sb:
        return []
    try:
        res = safe_execute(
            sb.table(TABLE).select("*").order("position_slot").order("rank")
        )
        return res.data or []
    except Exception as e:
        print(f"[shadow_team_repo] ❌ list_shadow_team failed: {e}")
        return []


def delete_shadow_team(player: str) -> bool:
    sb = get_supabase_client()
    if not sb:
        return False
    try:
        safe_execute(sb.table(TABLE).delete().eq("player", player))
        return True
    except Exception as e:
        print(f"[shadow_team_repo] ❌ delete_shadow_team failed for {player}: {e}")
        return False
