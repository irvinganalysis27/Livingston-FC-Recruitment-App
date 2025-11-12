# ============================================================
# ⭐ WATCH LIST PAGE (with radar-identical mapping + Shadow Team add)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
from auth import check_password
from branding import show_branding

# ========= Streamlit Config =========
st.set_page_config(page_title="Livingston FC Recruitment App", layout="centered")

# ========= Password =========

from auth import check_password
if not check_password():
    st.stop()

# ---------- Branding ----------
show_branding()
st.title("⭐ Watch List")

# --- Style (scrollable text areas) ---
st.markdown("""
<style>
textarea {
    resize: vertical !important;
    overflow-y: scroll !important;
    scrollbar-width: thin !important;
    scrollbar-color: #888 #f1f1f1 !important;
    min-height: 60px;
}
</style>
""", unsafe_allow_html=True)

# ========= External Links =========
st.markdown(
    """
    📊 [**Open Google Watch List Sheet**](https://docs.google.com/spreadsheets/d/1ESiZsk7W-LrotYs7hpznJB4K-dHgKA0bWE1oUCJ8Pf0/edit?gid=0#gid=0)
    """,
    unsafe_allow_html=True,
)
st.divider()

# ========= Toast helpers =========
def toast_ok(msg: str):
    st.toast(msg, icon="✅")

def toast_err(msg: str):
    st.toast(msg, icon="❌")

# ========= Imports from shared repo =========
from lib.favourites_repo import (
    list_favourites,
    upsert_favourite,
    delete_favourite,
    hide_favourite,
)

# ✅ Import new shadow team repo
from lib.shadow_team_repo import upsert_shadow_team

# ========= Domain constants =========
COLOUR_CHOICES = [
    "🟣 Needs Checked",
    "🟡 Monitor",
    "🟢 Go",
    "🟠 Out Of Reach",
    "🔴 No Further Interest",
]
COLOUR_EMOJI = {c.split(" ", 1)[0]: c for c in COLOUR_CHOICES}

SIX_GROUPS = [
    "Full Back", "Centre Back", "Number 6", "Number 8", "Winger", "Striker"
]

# ========= Position Mapping (EXACT radar logic) =========
def _clean_pos_token(tok: str) -> str:
    if pd.isna(tok):
        return ""
    t = str(tok).upper().strip()
    t = re.sub(r"[.\-_/]", " ", t)
    t = re.sub(r"\s+", "", t)
    return t

RAW_TO_GROUP = {
    "RIGHTBACK": "Full Back", "LEFTBACK": "Full Back",
    "RIGHTWINGBACK": "Full Back", "LEFTWINGBACK": "Full Back",
    "CENTREBACK": "Centre Back",
    "RIGHTCENTREBACK": "Centre Back", "LEFTCENTREBACK": "Centre Back",
    "DEFENSIVEMIDFIELDER": "Number 6",
    "RIGHTDEFENSIVEMIDFIELDER": "Number 6",
    "LEFTDEFENSIVEMIDFIELDER": "Number 6",
    "CENTREDEFENSIVEMIDFIELDER": "Number 6",
    "CENTREMIDFIELDER": "Number 8",
    "LEFTCENTREMIDFIELDER": "Number 8",
    "RIGHTCENTREMIDFIELDER": "Number 8",
    "CENTREATTACKINGMIDFIELDER": "Number 8",
    "RIGHTATTACKINGMIDFIELDER": "Number 8",
    "LEFTATTACKINGMIDFIELDER": "Number 8",
    "LEFTWING": "Winger", "RIGHTWING": "Winger",
    "LEFTMIDFIELDER": "Winger", "RIGHTMIDFIELDER": "Winger",
    "CENTREFORWARD": "Striker",
    "LEFTCENTREFORWARD": "Striker",
    "RIGHTCENTREFORWARD": "Striker",
    "SECONDSTRIKER": "Striker",
    "10": "Striker",
}

def map_first_position_to_group(primary_pos_cell: str) -> str:
    if pd.isna(primary_pos_cell) or not str(primary_pos_cell).strip():
        return None
    parts = re.split(r"[,/]", str(primary_pos_cell))
    for p in parts:
        tok = _clean_pos_token(p)
        if tok in RAW_TO_GROUP:
            return RAW_TO_GROUP[tok]
    return None

# ========= Page controls / filters =========
top_c1, top_c2 = st.columns([1, 1])
with top_c1:
    show_hidden = st.toggle("Show hidden players", value=False)
with top_c2:
    st.caption("Filter by colour/status or hide players below.")

if "status_filter" not in st.session_state:
    st.session_state.status_filter = ["🟣 Needs Checked"]

selected_statuses = st.multiselect(
    "Filter by Status",
    options=COLOUR_CHOICES,
    default=st.session_state.status_filter,
    key="status_filter",
    label_visibility="collapsed"
)

rows_all = list_favourites(only_visible=not show_hidden)
for r in rows_all:
    raw_pos = r.get("position", "")
    r["mapped_position"] = map_first_position_to_group(raw_pos) or raw_pos

available_groups = sorted({
    r["mapped_position"] for r in rows_all if r.get("mapped_position")
})
available_groups = [g for g in SIX_GROUPS if g in available_groups]

if "fav_pos_groups" not in st.session_state:
    st.session_state.fav_pos_groups = available_groups.copy()

selected_groups = st.multiselect(
    "Filter by Position Group",
    options=available_groups,
    default=st.session_state.fav_pos_groups,
    key="fav_pos_group_multiselect",
    label_visibility="collapsed"
)

rows = [r for r in rows_all if r.get("colour") in selected_statuses]
if selected_groups and len(selected_groups) < len(available_groups):
    rows = [r for r in rows if r.get("mapped_position") in selected_groups]


# ============================================================
# ➕ ADD NEW PLAYER MANUALLY
# ============================================================
with st.expander("➕ Add New Player to Favourites", expanded=False):
    st.markdown("Use this form to manually add a new player record.")

    c1, c2 = st.columns(2)
    with c1:
        new_player = st.text_input("Player Name*", key="new_player_name")
        new_team = st.text_input("Team", key="new_player_team")
        new_league = st.text_input("League", key="new_player_league")

    with c2:
        new_position = st.selectbox(
            "Position Group",
            ["Full Back", "Centre Back", "Number 6", "Number 8", "Winger", "Striker"],
            key="new_player_pos"
        )

    c3, _ = st.columns([1, 4])
    with c3:
        if st.button("💾 Create Player", key="create_new_player"):
            if not new_player.strip():
                st.warning("⚠️ Please enter a player name before creating a record.")
            else:
                payload = {
                    "player": new_player.strip(),
                    "team": new_team.strip(),
                    "league": new_league.strip(),
                    "position": new_position.strip(),
                    "colour": "🟣 Needs Checked",
                    "initial_watch_comment": "",
                    "second_watch_comment": "",
                    "latest_action": "",
                    "visible": True,
                    "updated_by": st.session_state.get("user_initials", ""),
                    "source": "manual-add",
                }

                ok = upsert_favourite(payload, log_to_sheet=True)
                if ok:
                    toast_ok(f"✅ Added {new_player} to favourites.")
                    st.rerun()
                else:
                    toast_err(f"❌ Failed to add {new_player}.")

# ============================================================
# 🧾 Render current list (with Shadow Team add)
# ============================================================
if not rows:
    st.info("No favourites found for the selected filters.")
else:
    for row in rows:
        with st.container(border=True):
            player = row.get("player", "")
            team = row.get("team", "") or ""
            league = row.get("league", "") or ""
            position = row.get("mapped_position", row.get("position", "") or "")

            st.markdown(
                f"**{player}** &nbsp;&nbsp; "
                f"<span style='opacity:0.7'>{team or '—'}, {league or '—'}, {position or '—'}</span>",
                unsafe_allow_html=True,
            )

            # --- Colour + Comments ---
            c1, c2 = st.columns([1, 2])
            with c1:
                current_colour = row.get("colour") or ""
                colour_choice = st.selectbox(
                    "Status",
                    options=COLOUR_CHOICES,
                    index=COLOUR_CHOICES.index(current_colour) if current_colour in COLOUR_CHOICES else 0,
                    key=f"colour_{player}",
                )
            with c2:
                initial_comment = st.text_area(
                    "Initial Watch",
                    value=row.get("initial_watch_comment") or "",
                    key=f"initial_{player}",
                    height=160,
                )

            _, c2b = st.columns([1, 2])
            with c2b:
                second_comment = st.text_area(
                    "Second Watch",
                    value=row.get("second_watch_comment") or "",
                    key=f"second_{player}",
                    height=160,
                )

            latest_action = st.text_area(
                "Latest Action",
                value=row.get("latest_action") or "",
                key=f"latest_{player}",
                height=60,
            )

            c3, c4, c5, c6 = st.columns([0.4, 0.2, 0.2, 0.2])
            with c3:
                visible_val = st.checkbox("Visible", value=bool(row.get("visible", True)), key=f"vis_{player}")

            with c4:
                if st.button("💾 Save", key=f"save_{player}"):
                    payload = {
                        "player": player,
                        "team": team,
                        "league": league,
                        "position": position,
                        "colour": colour_choice,
                        "initial_watch_comment": initial_comment,
                        "second_watch_comment": second_comment,
                        "latest_action": latest_action,
                        "visible": visible_val,
                        "updated_by": st.session_state.get("user_initials", ""),
                        "source": "watchlist-page",
                    }
                    ok = upsert_favourite(payload, log_to_sheet=True)
                    if ok:
                        toast_ok(f"Saved changes for {player}")
                        st.rerun()
                    else:
                        toast_err(f"Failed to save {player}")

            with c5:
                if st.button("🗑 Remove", key=f"del_{player}"):
                    if delete_favourite(player):
                        toast_ok(f"Removed {player}")
                        st.rerun()
                    else:
                        toast_err(f"Failed to remove {player}")

            # ✅ NEW: Add to Shadow Team
            with c6:
                # Make button fill width
                st.markdown(
                    """
                    <style>
                    div[data-testid="stButton"] button[kind="secondary"] {
                        width: 100% !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                # Main button
                if st.button("➕ Shadow Team", key=f"shadow_{player}", width="stretch"):
                    st.session_state[f"show_popover_{player}"] = True  # open manually

                # Only show popover when clicked
                if st.session_state.get(f"show_popover_{player}", False):
                    with st.popover(f"Add {player} to Shadow Team", width="stretch"):
                        # Store position in session_state so it persists through reruns
                        pos_key = f"shadow_pos_{player}"
                        if pos_key not in st.session_state:
                            st.session_state[pos_key] = "ST"

                        selected_pos = st.selectbox(
                            "Position slot",
                            ["GK", "RB", "LB", "CB", "6", "8", "RW", "LW", "ST"],
                            index=["GK", "RB", "LB", "CB", "6", "8", "RW", "LW", "ST"].index(
                                st.session_state.get(pos_key, "ST")
                            ),
                            key=pos_key,
                        )

                        # Confirm button inside popover
                        if st.button("✅ Confirm", key=f"shadow_add_{player}"):
                            pos_slot = st.session_state[pos_key]
                            payload = {"player": player, "position_slot": pos_slot, "rank": 0}
                            ok = upsert_shadow_team(payload)
                            if ok:
                                toast_ok(f"✅ Added {player} to Shadow Team as {pos_slot}")
                            else:
                                toast_err("❌ Failed to add player to Shadow Team")
                            st.session_state[f"show_popover_{player}"] = False  # close after adding
