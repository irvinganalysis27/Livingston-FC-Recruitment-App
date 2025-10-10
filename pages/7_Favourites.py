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
# 🔒 Password protection
# ============================================================
if not check_password():
    st.stop()

show_branding()
st.title("⭐ Watch List")

# Link to shared Google Sheet
SHEET_LINK = "https://docs.google.com/spreadsheets/d/16oweZkbqNst16U5lQshnYjwjWiuCmd8ClZfPyAtoE0o/edit?usp=sharing"
st.markdown(f"📊 [**Open Livingston Favourites Google Sheet**]({SHEET_LINK})")

DB_PATH = Path(__file__).parent / "favourites.db"

# ============================================================
# 🧱 Database init
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
# 📄 Google Sheets connection
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
        return client.open("Livingston_Favourites_Log").sheet1
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
    except Exception as e:
        st.error(f"❌ Failed to log {player}: {e}")

# ============================================================
# 🧩 Auto-restore from Google Sheet if local DB is empty
# ============================================================
def restore_if_empty():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    count = c.execute("SELECT COUNT(*) FROM favourites").fetchone()[0]
    conn.close()

    if count > 0:
        return

    st.warning("📥 Local database empty, restoring from Google Sheet…")
    sheet = init_sheet()
    if not sheet:
        st.error("❌ Could not connect to Google Sheet for restore.")
        return

    try:
        rows = sheet.get_all_records()
        if not rows:
            st.info("Google Sheet is empty, nothing to restore.")
            return

        df = pd.DataFrame(rows)
        keep_cols = ["player", "team", "league", "position", "colour", "comment"]
        df = df.rename(columns=str.lower)[keep_cols]
        df["visible"] = 1
        df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        conn = sqlite3.connect(DB_PATH)
        df.to_sql("favourites", conn, if_exists="replace", index=False)
        conn.close()

        st.success(f"✅ Restored {len(df)} favourites from Google Sheet")
    except Exception as e:
        st.error(f"❌ Restore failed: {e}")

restore_if_empty()

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
# 🧠 Page layout
# ============================================================
show_hidden = st.checkbox("Show hidden players", value=False)
rows = get_favourites(show_hidden)

if not rows:
    st.info("No favourites saved yet.")
    st.stop()

st.markdown("""
**How to use this list:**
- 🟢 **Choose Colour:** set a status for each player.  
- 💬 **Write Comment:** add your initials and scouting notes.  
- 👁️ **Deselect "Visible":** to hide completed players.  
- 🗑️ **Added a player by accident?** Tick **Remove** to delete completely.  
""")

df = pd.DataFrame(rows, columns=["Player", "Team", "League", "Position", "Colour", "Comment", "Visible", "Timestamp"])
df["Remove"] = False

colour_options = ["🟣 Needs Checked", "🟡 Monitor", "🟢 Go", "🔴 No Further Interest"]

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
# 💾 Apply updates
# ============================================================
removed_players, logged_changes = [], 0

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

st.info(f"Logged {logged_changes} change(s). Removed {len(removed_players)} player(s).")
