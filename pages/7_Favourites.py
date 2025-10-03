# pages/7_Favourites.py

import streamlit as st
import sqlite3
from pathlib import Path
from auth import check_password
from branding import show_branding
from datetime import datetime

import gspread
from google.oauth2.service_account import Credentials

# ============================================================
# Protect page
# ============================================================
if not check_password():
    st.stop()

show_branding()
st.title("⭐ Favourite Players")

# ============================================================
# Database setup (with migration)
# ============================================================
DB_PATH = Path(__file__).parent / "favourites.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Create table if not exists
    c.execute("""
        CREATE TABLE IF NOT EXISTS favourites (
            player TEXT PRIMARY KEY,
            team TEXT,
            league TEXT,
            position TEXT,
            colour TEXT DEFAULT 'Yellow',
            comment TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Migration: ensure new columns exist
    existing_cols = [row[1] for row in c.execute("PRAGMA table_info(favourites)").fetchall()]

    if "colour" not in existing_cols:
        c.execute("ALTER TABLE favourites ADD COLUMN colour TEXT DEFAULT 'Yellow'")
    if "comment" not in existing_cols:
        c.execute("ALTER TABLE favourites ADD COLUMN comment TEXT")
    if "timestamp" not in existing_cols:
        c.execute("ALTER TABLE favourites ADD COLUMN timestamp DATETIME DEFAULT CURRENT_TIMESTAMP")

    conn.commit()
    conn.close()

init_db()

# ============================================================
# Google Sheets setup
# ============================================================
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

def init_sheet():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=SCOPES
    )
    client = gspread.authorize(creds)
    sheet = client.open("Livingston_Favourites_Log").sheet1
    return sheet

def log_to_sheet(player, team, league, position, colour, comment, action):
    try:
        sheet = init_sheet()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([player, team, league, position, colour, comment, action, now])
    except Exception as e:
        st.error(f"❌ Failed to log to Google Sheet: {e}")

# ============================================================
# Database functions
# ============================================================
def get_favourites():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT player, team, league, position, colour, comment, timestamp FROM favourites ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return rows

def add_or_update_favourite(player, team, league, position, colour="Yellow", comment=""):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO favourites (player, team, league, position, colour, comment, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (player, team, league, position, colour, comment))
    conn.commit()
    conn.close()
    log_to_sheet(player, team, league, position, colour, comment, "Added/Updated")

def remove_favourite(player):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT team, league, position, colour, comment FROM favourites WHERE player=?", (player,))
    row = c.fetchone()
    c.execute("DELETE FROM favourites WHERE player=?", (player,))
    conn.commit()
    conn.close()
    if row:
        team, league, position, colour, comment = row
        log_to_sheet(player, team, league, position, colour, comment, "Removed")

# ============================================================
# Helper: Colour tags
# ============================================================
COLOUR_OPTIONS = {
    "Green": "🟢 Go",
    "Yellow": "🟡 Monitor",
    "Red": "🔴 No Further Interest"
}

def colour_tag(colour: str) -> str:
    mapping = {
        "Green": '<span style="color:green; font-weight:bold;">🟢 Go</span>',
        "Yellow": '<span style="color:orange; font-weight:bold;">🟡 Monitor</span>',
        "Red": '<span style="color:red; font-weight:bold;">🔴 No Further Interest</span>'
    }
    return mapping.get(colour, colour)

# ============================================================
# UI
# ============================================================
favs = get_favourites()

if favs:
    st.markdown("### Your Favourites")

    for player, team, league, position, colour, comment, ts in favs:
        with st.container():
            col1, col2, col3 = st.columns([4, 2, 1])
            with col1:
                st.markdown(
                    f"**{player}** | {team} | {league} | {position} | {colour_tag(colour)}",
                    unsafe_allow_html=True
                )
                new_comment = st.text_input(f"Comment for {player}", value=comment if comment else "", key=f"comment_{player}")
            with col2:
                new_colour_label = st.selectbox(
                    "Status",
                    list(COLOUR_OPTIONS.values()),
                    index=list(COLOUR_OPTIONS.keys()).index(colour if colour in COLOUR_OPTIONS else "Yellow"),
                    key=f"colour_{player}"
                )
                # Map back to plain database value
                new_colour = [k for k, v in COLOUR_OPTIONS.items() if v == new_colour_label][0]
            with col3:
                if st.button("❌ Remove", key=f"remove_{player}"):
                    remove_favourite(player)
                    st.success(f"Removed {player} from favourites")
                    st.rerun()

            # Save updates if anything changed
            if new_comment != (comment or "") or new_colour != (colour or "Yellow"):
                add_or_update_favourite(player, team, league, position, new_colour, new_comment)
                st.rerun()

else:
    st.info("No favourites have been added yet.")
