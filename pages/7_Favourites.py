import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
from auth import check_password
from branding import show_branding
import gspread
from google.oauth2.service_account import Credentials

# ============================================================
# Protect page
# ============================================================
if not check_password():
    st.stop()

show_branding()
st.title("⭐ Favourite Players")

DB_PATH = Path(__file__).parent / "favourites.db"

# ============================================================
# DB init
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
# Google Sheets
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
        st.write(f"✅ Logged to sheet: {player} → {action}")
    except Exception as e:
        st.error(f"❌ Failed to log {player}: {e}")

# ============================================================
# DB ops
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
# Table logic
# ============================================================
show_hidden = st.checkbox("Show hidden players", value=False)
rows = get_favourites(show_hidden)
if not rows:
    st.info("No favourites saved yet.")
    st.stop()

df = pd.DataFrame(rows, columns=["Player", "Team", "League", "Position", "Colour", "Comment", "Visible", "Timestamp"])
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
# Apply changes
# ============================================================
removed_players = []
logged_changes = 0

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

    if remove_flag:
        delete_favourite(player)
        log_to_sheet(player, row["Team"], row["League"], row["Position"], colour, comment, "Removed")
        st.write(f"🗑️ Removed {player}")
        removed_players.append(player)
        st.rerun()

    update_favourite(player, colour, comment, visible)

    if changed:
        action = "Hidden" if visible == 0 else "Updated"
        st.write(f"🟨 Change detected for {player}: {action}")
        log_to_sheet(player, row["Team"], row["League"], row["Position"], colour, comment, action)
        logged_changes += 1

# ============================================================
# Summary
# ============================================================
st.info(f"Logged {logged_changes} change(s). Removed {len(removed_players)} player(s).")
