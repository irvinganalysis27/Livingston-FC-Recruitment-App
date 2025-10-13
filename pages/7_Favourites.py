# pages/3_Favourites.py  (DEV app)
import streamlit as st
import pandas as pd
from lib.favourites_repo import list_favourites, upsert_favourite, set_visible, delete_favourite
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials

st.title("⭐ Watch List (DEV)")

# --- Load existing
rows = list_favourites(only_visible=False)
df = pd.DataFrame(rows)

# --- Left: current list
st.subheader("Current")
if df.empty:
    st.info("No favourites yet.")
else:
    st.dataframe(
        df[["player","team","league","position","colour","comment","visible","updated_at"]]
        .sort_values(["visible","player"], ascending=[False, True]),
        use_container_width=True
    )

# --- Right: editor
st.subheader("Add / Edit")
with st.form("fav_form", clear_on_submit=False):
    player = st.text_input("Player*", "")
    team = st.text_input("Team", "")
    league = st.text_input("League", "")
    position = st.text_input("Position", "")
    colour = st.selectbox("Status / Colour", ["", "🟢 Go", "🟡 Monitor", "🟣 Needs Checked", "🔴 No Further Interest"])
    comment = st.text_area("Comment", "")
    visible = st.checkbox("Visible", value=True)
    updated_by = st.text_input("Your initials (optional)", "")

    submitted = st.form_submit_button("💾 Save Changes")
    if submitted:
        if not player.strip():
            st.error("Player name is required.")
        else:
            upsert_favourite(
                player=player.strip(),
                team=team.strip(),
                league=league.strip(),
                position=position.strip(),
                colour=colour.strip(),
                comment=comment.strip(),
                visible=visible,
                updated_by=updated_by.strip() or None,
                source="dev"
            )
            st.success(f"Saved '{player}'.")
            st.rerun()

# --- Quick actions
st.subheader("Quick Actions")
col1, col2, col3 = st.columns(3)
with col1:
    name_hide = st.text_input("Hide player", key="hide_name")
    if st.button("Hide", type="secondary"):
        if name_hide.strip():
            set_visible(name_hide.strip(), False)
            st.rerun()
with col2:
    name_show = st.text_input("Show player", key="show_name")
    if st.button("Show", type="secondary"):
        if name_show.strip():
            set_visible(name_show.strip(), True)
            st.rerun()
with col3:
    name_del = st.text_input("Delete player", key="delete_name")
    if st.button("Delete", type="primary"):
        if name_del.strip():
            delete_favourite(name_del.strip())
            st.rerun()

# --- Google Sheet log (optional; runs on save)
def append_log_to_sheet(entry: dict):
    try:
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"],
                    scopes=["https://www.googleapis.com/auth/spreadsheets"])
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(st.secrets.get("SHEET_ID", ""))  # put your Sheet ID in secrets if you want
        ws = sh.worksheet("favourites_log")  # create if missing
        ws.append_row([
            entry.get("player",""),
            entry.get("league",""),
            entry.get("team",""),
            entry.get("position",""),
            entry.get("colour",""),
            entry.get("comment",""),
            entry.get("visible", True),
            entry.get("updated_by",""),
            datetime.utcnow().isoformat(timespec="seconds")+"Z",
        ])
    except Exception as e:
        st.caption(f"Log to Sheet skipped: {e}")
