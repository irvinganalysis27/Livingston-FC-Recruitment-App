# ============================================================
# 🧩 SHADOW TEAM REPOSITORY
# ============================================================

from datetime import datetime, timezone
import time
import streamlit as st
from supabase import create_client

TABLE = "shadow_team"

# ============================================================
# 🔗 Supabase Connection (fresh client per call)
# ============================================================

def get_supabase_client():
    """Return a new Supabase client using the service key."""
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["service_key"]  # ✅ use service role key
        return create_client(url, key)
    except Exception as e:
        st.error(f"❌ Supabase connection failed: {e}")
        return None


# ============================================================
# 🛡️ Safe Execute Helper
# ============================================================

def safe_execute(query, retries=3, delay=0.3):
    """Retry Supabase queries if temporary connection issues occur."""
    for i in range(retries):
        try:
            return query.execute()
        except Exception as e:
            if "Resource temporarily unavailable" in str(e) and i < retries - 1:
                time.sleep(delay)
                continue
            raise


# ============================================================
# 💾 CRUD FUNCTIONS
# ============================================================

def upsert_shadow_team(record: dict) -> bool:
    """
    Insert or update a player in the Shadow Team table.
    record = {"player": str, "position_slot": str, "rank": int}
    """
    sb = get_supabase_client()
    if not sb:
        return False

    player = (record.get("player") or "").strip()
    if not player:
        print("[shadow_team_repo] ⚠️ No player provided.")
        return False

    payload = {
        "player": player,
        "position_slot": record.get("position_slot") or "ST",
        "rank": int(record.get("rank") or 0),
        "notes": record.get("notes") or "",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        # Check if record already exists
        existing = sb.table(TABLE).select("*") \
            .eq("player", player) \
            .eq("position_slot", payload["position_slot"]) \
            .execute()
        print(f"[shadow_team_repo] Existing check for {player}: {existing.data}")

        if existing.data:
            # Update
            res = safe_execute(
                sb.table(TABLE).update(payload).eq("player", player)
            )
            print(f"[shadow_team_repo] ✅ Updated record for {player}: {res.data}")
        else:
            # Insert (will upsert by player)
            res = safe_execute(
                sb.table(TABLE).upsert(payload, on_conflict="player,position_slot")
            )
            print(f"[shadow_team_repo] ✅ Inserted record for {player}: {res.data}")

        return True

    except Exception as e:
        print(f"[shadow_team_repo] ❌ Upsert failed for {player}: {e}")
        st.error(f"Shadow Team upsert failed for {player}: {e}")
        return False


def list_shadow_team():
    """List all shadow team records, ordered by position then rank."""
    sb = get_supabase_client()
    if not sb:
        return []
    try:
        res = safe_execute(
            sb.table(TABLE).select("*").order("position_slot").order("rank")
        )
        print(f"[shadow_team_repo] ✅ Retrieved {len(res.data or [])} shadow team rows")
        return res.data or []
    except Exception as e:
        print(f"[shadow_team_repo] ❌ list_shadow_team failed: {e}")
        return []


def delete_shadow_team(player: str) -> bool:
    """Delete a player from the Shadow Team."""
    sb = get_supabase_client()
    if not sb:
        return False
    try:
        safe_execute(sb.table(TABLE).delete().eq("player", player))
        print(f"[shadow_team_repo] 🗑 Deleted from shadow team: {player}")
        return True
    except Exception as e:
        print(f"[shadow_team_repo] ❌ delete_shadow_team failed for {player}: {e}")
        return False
