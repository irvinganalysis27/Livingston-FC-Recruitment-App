import os
import re
from datetime import datetime

import streamlit as st
from supabase import create_client, Client

# ========= Basic page chrome =========
st.set_page_config(page_title="⭐ Watch List", layout="centered")
st.title("⭐ Watch List (DEV)")

# Small helper for consistent toasts
def toast_ok(msg: str):
    st.toast(msg, icon="✅")

def toast_err(msg: str):
    st.toast(msg, icon="❌")


# ========= Supabase client =========
@st.cache_resource(show_spinner=False)
def get_sb() -> Client | None:
    # Read from Streamlit secrets
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["service_key"]  # must be the SERVICE ROLE key
    except Exception:
        st.error("Supabase secrets not found. Add them under Settings → Secrets.\n"
                 "Expected [supabase] url + service_key.")
        return None

    if not url or not key:
        st.error("Supabase url / service_key is missing in secrets. Please add your SERVICE ROLE key.")
        return None

    try:
        return create_client(url, key)
    except Exception as e:
        st.error(f"Failed to initialise Supabase client: {e}")
        return None


sb = get_sb()
TABLE = "favourites"

# ========= Domain constants =========
COLOUR_CHOICES = [
    "🟣 Needs Checked",
    "🟡 Monitor",
    "🟢 Go",
    "🔴 No Further Interest",
]
COLOUR_EMOJI = {c.split(" ", 1)[0]: c for c in COLOUR_CHOICES}  # "🟡" -> "🟡 Monitor"


# ========= Repository helpers =========
def list_favourites(only_visible: bool = True) -> list[dict]:
    """Return list of favourites (dict rows), optionally only visible ones."""
    if sb is None:
        return []

    q = sb.table(TABLE).select("*").order("player", desc=False)
    if only_visible:
        q = q.eq("visible", True)
    try:
        res = q.execute()
        return res.data or []
    except Exception as e:
        toast_err(f"Read error: {e}")
        return []


def upsert_favourite(
    player: str,
    team: str = "",
    league: str = "",
    position: str = "",
    colour: str = "",
    comment: str = "",
    visible: bool = True,
    updated_by: str | None = None,
    source: str | None = "dev",
) -> bool:
    """Insert or update a favourite row (player is the PK)."""
    if sb is None:
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
        "source": source or "",
    }

    try:
        sb.table(TABLE).upsert(payload, on_conflict="player").execute()
        return True
    except Exception as e:
        toast_err(f"Save failed for {player}: {e}")
        return False


def update_visible(player: str, visible: bool) -> bool:
    if sb is None:
        return False
    try:
        sb.table(TABLE).update({"visible": bool(visible), "updated_at": datetime.utcnow().isoformat()}).eq("player", player).execute()
        return True
    except Exception as e:
        toast_err(f"Visibility update failed: {e}")
        return False


def delete_favourite(player: str) -> bool:
    if sb is None:
        return False
    try:
        sb.table(TABLE).delete().eq("player", player).execute()
        return True
    except Exception as e:
        toast_err(f"Delete failed: {e}")
        return False


# ========= Page controls / filters =========
top_c1, top_c2 = st.columns([1, 1])
with top_c1:
    show_hidden = st.toggle("Show hidden players", value=False)
with top_c2:
    st.write("")  # spacing
    st.caption("Choose a colour/status, add a short comment, and toggle visibility. Nothing is saved until you click **Save** on a card.")

rows = list_favourites(only_visible=not show_hidden)

# ========= Render current list (cards) =========
if not rows:
    st.info("No favourites yet.")
else:
    for row in rows:
        with st.container(border=True):
            player = row.get("player", "")
            team = row.get("team", "") or ""
            league = row.get("league", "") or ""
            position = row.get("position", "") or ""

            # Top line: player + meta
            st.markdown(
                f"**{player}** &nbsp;&nbsp; "
                f"<span style='opacity:0.7'>{team or '—'}, {league or '—'}, {position or '—'}</span>",
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns([1, 2])
            with c1:
                # status / colour
                current_colour = row.get("colour") or ""
                if current_colour not in COLOUR_CHOICES and current_colour in COLOUR_EMOJI:
                    # if only emoji was stored, map to full label
                    current_colour = COLOUR_EMOJI[current_colour]
                colour_choice = st.selectbox(
                    "Status",
                    options=COLOUR_CHOICES,
                    index=COLOUR_CHOICES.index(current_colour) if current_colour in COLOUR_CHOICES else 1,
                    key=f"colour_{player}",
                )

            with c2:
                comment_val = st.text_input(
                    "Comment",
                    value=row.get("comment") or "",
                    key=f"comment_{player}",
                    placeholder="Initials + short note…",
                )

            c3, c4, c5 = st.columns([0.5, 0.25, 0.25])
            with c3:
                visible_val = st.checkbox("Visible", value=bool(row.get("visible", True)), key=f"vis_{player}")
            with c4:
                if st.button("💾 Save", key=f"save_{player}"):
                    ok = upsert_favourite(
                        player=player,
                        team=team,
                        league=league,
                        position=position,
                        colour=colour_choice,
                        comment=comment_val,
                        visible=visible_val,
                        updated_by=st.session_state.get("user_initials", ""),  # optional, add your own capture
                        source="dev",
                    )
                    if ok:
                        toast_ok(f"Saved changes for {player}")
                        st.rerun()
            with c5:
                if st.button("🗑 Remove", key=f"del_{player}"):
                    if delete_favourite(player):
                        toast_ok(f"Removed {player}")
                        st.rerun()


st.divider()

# ========= Add / Edit form =========
st.subheader("Add / Edit")

with st.form(key="add_edit_form", border=True):
    p = st.text_input("Player*", placeholder="Full name (must be unique)")
    t = st.text_input("Team", placeholder="e.g., Crotone")
    l = st.text_input("League", placeholder="e.g., Italy Serie C")
    pos = st.text_input("Position", placeholder="e.g., Right Wing, Centre Forward")
    col = st.selectbox("Status / Colour", COLOUR_CHOICES, index=1)
    com = st.text_area("Comment", placeholder="Initials + short note…")
    vis = st.checkbox("Visible", value=True)
    initials = st.text_input("Your initials (optional)")

    bcol1, bcol2 = st.columns([0.2, 0.8])
    with bcol1:
        submit = st.form_submit_button("💾 Save Changes")

    if submit:
        player_clean = (p or "").strip()
        if not player_clean:
            toast_err("Player is required.")
        else:
            ok = upsert_favourite(
                player=player_clean,
                team=t.strip(),
                league=l.strip(),
                position=pos.strip(),
                colour=col,
                comment=com.strip(),
                visible=vis,
                updated_by=initials.strip(),
                source="dev",
            )
            if ok:
                toast_ok(f"Saved {player_clean}")
                st.rerun()

# ========= Quick actions =========
st.subheader("Quick Actions")

qa1, qa2, qa3 = st.columns(3)

with qa1:
    hide_name = st.text_input("Hide player", key="hide_in")
    if st.button("Hide"):
        if hide_name.strip():
            if update_visible(hide_name.strip(), False):
                toast_ok(f"Hid {hide_name.strip()}")
                st.rerun()

with qa2:
    show_name = st.text_input("Show player", key="show_in")
    if st.button("Show"):
        if show_name.strip():
            if update_visible(show_name.strip(), True):
                toast_ok(f"Made {show_name.strip()} visible")
                st.rerun()

with qa3:
    del_name = st.text_input("Delete player", key="del_in")
    if st.button("Delete"):
        if del_name.strip():
            if delete_favourite(del_name.strip()):
                toast_ok(f"Deleted {del_name.strip()}")
                st.rerun()
