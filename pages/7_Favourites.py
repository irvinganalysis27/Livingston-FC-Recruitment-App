import streamlit as st
import sqlite3
import pandas as pd
import json
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
QUEUE_FILE = Path(__file__).parent / "pending_logs.json"

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
# 📄 Google Sheets logging setup
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
    except Exception:
        return None

# ============================================================
# 🧰 Queue handling (persistent async logging)
# ============================================================
def load_queue():
    if QUEUE_FILE.exists():
        try:
            return json.loads(QUEUE_FILE.read_text())
        except Exception:
            return []
    return []

def save_queue(queue):
    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f)

def enqueue_log(player, team, league, position, colour, comment, action):
    queue = load_queue()
    queue.append({
        "player": player,
        "team": team,
        "league": league,
        "position": position,
        "colour": colour,
        "comment": comment,
        "action": action,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_queue(queue)

def flush_logs():
    queue = load_queue()
    if not queue:
        return

    sheet = init_sheet()
    if not sheet:
        return  # keep queue until next rerun if sheet unavailable

    new_queue = []
    for entry in queue:
        try:
            sheet.append_row([
                entry["player"],
                entry["team"],
                entry["league"],
                entry["position"],
                entry["colour"],
                entry["comment"],
                entry["action"],
                entry["timestamp"],
            ])
        except Exception:
            new_queue.append(entry)  # retry next run

    save_queue(new_queue)

# Run queued logs first (non-blocking)
flush_logs()

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
# 💾 Apply changes (instant save + async logging)
# ============================================================
removed_players = []
logged_changes = 0
status_messages = []

if "last_saved" not in st.session_state:
    st.session_state["last_saved"] = {}

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

    prev_key = f"{player}_{colour}_{comment}_{visible}_{remove_flag}"
    if st.session_state["last_saved"].get(player) == prev_key:
        continue
    st.session_state["last_saved"][player] = prev_key

    if remove_flag:
        delete_favourite(player)
        enqueue_log(player, row["Team"], row["League"], row["Position"], colour, comment, "Removed")
        status_messages.append(f"🗑️ {player} permanently removed from list")
        removed_players.append(player)
        st.session_state["needs_rerun"] = True
        continue

    if changed:
        update_favourite(player, colour, comment, visible)
        if int(prev["Visible"]) != visible and visible == 0:
            msg, action = f"👁️ {player} hidden from list", "Hidden"
        elif comment != prev["Comment"]:
            msg, action = f"💬 Comment saved for {player}", "Comment Updated"
        elif colour != prev["Colour"]:
            msg, action = f"✅ Status saved for {player}", "Status Updated"
        else:
            msg, action = f"💾 Changes saved for {player}", "Updated"

        enqueue_log(player, row["Team"], row["League"], row["Position"], colour, comment, action)
        status_messages.append(msg)
        logged_changes += 1

if status_messages:
    st.success("\n".join(status_messages))

if st.session_state.get("needs_rerun", False):
    del st.session_state["needs_rerun"]
    st.rerun()
