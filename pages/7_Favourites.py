import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
from auth import check_password
from branding import show_branding
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(layout="wide")

# ============================================================
# 🔒 Protect page
# ============================================================
if not check_password():
    st.stop()

show_branding()
st.title("⭐ Watch List")

DB_PATH = Path(__file__).parent / "favourites.db"

# ============================================================
# 🧱 Database setup
# ============================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS favourites (
            player TEXT PRIMARY KEY,
            team TEXT,
            league TEXT,
            position TEXT,
            colour TEXT DEFAULT '',
            comment TEXT DEFAULT '',
            visible INTEGER DEFAULT 1,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ============================================================
# 📄 Google Sheets logging
# ============================================================
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

def init_sheet():
    try:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=SCOPES
        )
        client = gspread.authorize(creds)
        sheet = client.open("Livingston_Favourites_Log").sheet1
        return sheet
    except Exception as e:
        st.error(f"❌ Failed to connect to Google Sheets: {e}")
        return None

def log_to_sheet(player, team, league, position, colour, comment, action="Updated"):
    sheet = init_sheet()
    if not sheet:
        st.warning("⚠️ Skipped logging because sheet connection failed.")
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        sheet.append_row([player, team, league, position, colour, comment, action, now])
        st.caption(f"📋 {player} → {action} logged to sheet")
    except Exception as e:
        st.error(f"❌ Failed to log {player}: {e}")

# ============================================================
# ⚙️ Database operations
# ============================================================
def get_favourites(show_hidden=False):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    query = "SELECT player, team, league, position, colour, comment, visible, timestamp FROM favourites"
    if not show_hidden:
        query += " WHERE visible=1"
    query += " ORDER BY timestamp DESC"
    rows = c.execute(query).fetchall()
    conn.close()
    return rows

def update_favourite(player, colour, comment, visible):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        UPDATE favourites
        SET colour=?, comment=?, visible=?, timestamp=CURRENT_TIMESTAMP
        WHERE player=?
    """, (colour, comment, visible, player))
    conn.commit()
    conn.close()

def delete_favourite(player):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM favourites WHERE player=?", (player,))
    conn.commit()
    conn.close()

# ============================================================
# 🧩 Table logic
# ============================================================
show_hidden = st.checkbox("Show hidden players", value=False)
rows = get_favourites(show_hidden)

if not rows:
    st.info("No favourites saved yet.")
    st.stop()

st.markdown("""
**How to use this list:**
- 🟢 **Choose Colour:** Set a status for each player.  
- 💬 **Write Comment:** Add your initials and short scouting notes.  
- 👁️ **Deselect "Visible":** Hide the player when finished.  
- 🗑️ **Added a player by accident?** Tick **Remove** to delete permanently.  
""")

df = pd.DataFrame(rows, columns=[
    "Player", "Team", "League", "Position", "Colour", "Comment", "Visible", "Timestamp"
])
df["Remove"] = False

colour_options = [
    "🟣 Needs Checked",
    "🟡 Monitor",
    "🟢 Go",
    "🔴 No Further Interest"
]

st.markdown("### ✏️ Edit, Hide, or Remove Favourites")

edited_df = st.data_editor(
    df[["Player", "Team", "League", "Position", "Colour", "Comment", "Visible", "Remove"]],
    column_config={
        "Colour": st.column_config.SelectboxColumn("Colour", options=colour_options),
        "Comment": st.column_config.TextColumn("Comment"),
        "Visible": st.column_config.CheckboxColumn("Visible"),
        "Remove": st.column_config.CheckboxColumn("🗑️ Remove"),
    },
    hide_index=True,
    width="stretch",
)

# ============================================================
# 💾 Apply changes (single-run, debounced)
# ============================================================
removed_players = []
logged_changes = 0

# Store last saved values to prevent repeat triggers
if "last_saved" not in st.session_state:
    st.session_state["last_saved"] = {}

status_messages = []

for _, row in edited_df.iterrows():
    player = row["Player"]
    colour = row.get("Colour", "")
    comment = row.get("Comment", "")
    visible = int(row.get("Visible", True))
    remove_flag = bool(row.get("Remove", False))

    prev = df.loc[df["Player"] == player].iloc[0]
    changed = (
        (colour != prev["Colour"]) or
        (comment != prev["Comment"]) or
        (int(prev["Visible"]) != visible)
    )

    # --- Skip duplicate re-run noise ---
    prev_key = f"{player}_{colour}_{comment}_{visible}_{remove_flag}"
    if st.session_state["last_saved"].get(player) == prev_key:
        continue  # skip repeat of same change
    st.session_state["last_saved"][player] = prev_key

    # --- Handle removals ---
    if remove_flag:
        delete_favourite(player)
        log_to_sheet(player, row["Team"], row["League"], row["Position"], colour, comment, "Removed")
        status_messages.append(f"🗑️ {player} permanently removed from list")
        removed_players.append(player)
        st.session_state["needs_rerun"] = True
        continue

    # --- Save updates only if something changed ---
    if changed:
        update_favourite(player, colour, comment, visible)

        if int(prev["Visible"]) != visible and visible == 0:
            msg = f"👁️ {player} hidden from list"
            action = "Hidden"
        elif comment != prev["Comment"]:
            msg = f"💬 Comment saved for {player}"
            action = "Comment Updated"
        elif colour != prev["Colour"]:
            msg = f"✅ Status saved for {player}"
            action = "Status Updated"
        else:
            msg = f"💾 Changes saved for {player}"
            action = "Updated"

        status_messages.append(msg)
        log_to_sheet(player, row["Team"], row["League"], row["Position"], colour, comment, action)
        logged_changes += 1

# --- Show messages once per run ---
if status_messages:
    st.success("\n".join(status_messages))

# --- Single rerun after deletions only ---
if st.session_state.get("needs_rerun", False):
    del st.session_state["needs_rerun"]
    st.rerun()

# --- Summary ---
st.info(f"✅ Saved {logged_changes} change(s). Removed {len(removed_players)} player(s).")

# ============================================================
# 📊 Summary
# ============================================================
st.info(f"✅ Saved {logged_changes} change(s). Removed {len(removed_players)} player(s).")
