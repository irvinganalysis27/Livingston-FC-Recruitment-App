import streamlit as st
from datetime import datetime

# ========= Basic page chrome =========
st.set_page_config(page_title="⭐ Watch List", layout="centered")
st.title("⭐ Watch List")

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

# ========= Domain constants =========
COLOUR_CHOICES = [
    "🟣 Needs Checked",
    "🟡 Monitor",
    "🟢 Go",
    "🔴 No Further Interest",
]
COLOUR_EMOJI = {c.split(" ", 1)[0]: c for c in COLOUR_CHOICES}

# ========= Page controls / filters =========
top_c1, top_c2 = st.columns([1, 1])
with top_c1:
    show_hidden = st.toggle("Show hidden players", value=False)
with top_c2:
    st.caption("Filter by colour/status or hide players below.")

# --- NEW: Status filter ---
selected_statuses = st.multiselect(
    "Filter by Status",
    options=COLOUR_CHOICES,
    default=COLOUR_CHOICES,
    key="status_filter",
    label_visibility="collapsed"
)

# ========= Fetch and filter data =========
rows = list_favourites(only_visible=not show_hidden)

# Apply colour filter if selected
if selected_statuses and len(selected_statuses) < len(COLOUR_CHOICES):
    rows = [r for r in rows if r.get("colour") in selected_statuses]

# ========= Render current list (cards) =========
if not rows:
    st.info("No favourites found for the selected filters.")
else:
    for row in rows:
        with st.container(border=True):
            player = row.get("player", "")
            team = row.get("team", "") or ""
            league = row.get("league", "") or ""
            position = row.get("position", "") or ""

            # Top line
            st.markdown(
                f"**{player}** &nbsp;&nbsp; "
                f"<span style='opacity:0.7'>{team or '—'}, {league or '—'}, {position or '—'}</span>",
                unsafe_allow_html=True,
            )

            # --- Colour + Comments ---
            c1, c2 = st.columns([1, 2])
            with c1:
                current_colour = row.get("colour") or ""
                if current_colour not in COLOUR_CHOICES and current_colour in COLOUR_EMOJI:
                    current_colour = COLOUR_EMOJI[current_colour]
                colour_choice = st.selectbox(
                    "Status",
                    options=COLOUR_CHOICES,
                    index=COLOUR_CHOICES.index(current_colour) if current_colour in COLOUR_CHOICES else 0,
                    key=f"colour_{player}",
                )

            with c2:
                initial_comment = st.text_input(
                    "Initial Watch",
                    value=row.get("initial_watch_comment") or "",
                    key=f"initial_{player}",
                    placeholder="Initials + first comment…",
                )
                second_comment = st.text_input(
                    "Second Watch",
                    value=row.get("second_watch_comment") or "",
                    key=f"second_{player}",
                    placeholder="Initials + second comment…",
                )

            # --- Visibility + Actions ---
            c3, c4, c5 = st.columns([0.5, 0.25, 0.25])
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

st.divider()
