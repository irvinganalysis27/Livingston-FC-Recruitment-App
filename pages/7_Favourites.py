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
        return client.open("Livingston_Favourites_Log").sheet1
    except Exception:
        return None

def log_to_sheet(player, team, league, position, colour, comment, action="Updated"):
    sheet = init_sheet()
    if not sheet:
        st.warning("⚠️ Could not connect to Google Sheets.")
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        sheet.append_row([player, team, league, position, colour, comment, action, now])
    except Exception as e:
        st.error(f"❌ Failed to log {player}: {e}")

# ============================================================
# ⚙️ Database operations
# ============================================================
def get_favourites(show_hidden=False):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    query = "SELECT player, team, league, position, colour, comment, visible FROM favourites"
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
# 🧩 Page layout
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
- 👁️ **Deselect 'Visible':** Hide the player when finished.  
- 🗑️ **Added a player by accident?** Click **Remove** to delete permanently.  
""")

colour_options = ["🟣 Needs Checked", "🟡 Monitor", "🟢 Go", "🔴 No Further Interest"]

df = pd.DataFrame(rows, columns=["Player", "Team", "League", "Position", "Colour", "Comment", "Visible"])

# ============================================================
# 🧠 Per-row editing interface
# ============================================================
for _, row in df.iterrows():
    player, team, league, position, colour, comment, visible = row

    with st.container():
        st.markdown(f"### **{player}**  ({team}, {league}, {position})")

        col1, col2, col3, col4, col5 = st.columns([2, 3, 1, 1, 1])

        with col1:
            new_colour = st.selectbox(
                "Status",
                colour_options,
                index=colour_options.index(colour) if colour in colour_options else 1,
                key=f"colour_{player}"
            )

        with col2:
            new_comment = st.text_input(
                "Comment",
                value=comment if comment else "",
                key=f"comment_{player}"
            )

        with col3:
            new_visible = st.checkbox(
                "Visible",
                value=bool(visible),
                key=f"visible_{player}"
            )

        with col4:
            save = st.button("💾 Save", key=f"save_{player}")
        with col5:
            remove = st.button("❌ Remove", key=f"remove_{player}")

        # --- Handle actions ---
        if save:
            update_favourite(player, new_colour, new_comment, int(new_visible))
            log_to_sheet(player, team, league, position, new_colour, new_comment, "Updated")
            st.success(f"✅ Saved changes for {player}")

        if remove:
            delete_favourite(player)
            log_to_sheet(player, team, league, position, colour, comment, "Removed")
            st.error(f"🗑️ {player} removed from list")
            st.rerun()

        st.divider()
