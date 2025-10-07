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
# Initialise + migrate DB
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
    existing_cols = [row[1] for row in c.execute("PRAGMA table_info(favourites)").fetchall()]
    for col, dtype in [("colour", "TEXT DEFAULT ''"), ("comment", "TEXT DEFAULT ''"), ("visible", "INTEGER DEFAULT 1")]:
        if col not in existing_cols:
            c.execute(f"ALTER TABLE favourites ADD COLUMN {col} {dtype}")
    conn.commit()
    conn.close()

init_db()

# ============================================================
# Google Sheets Setup
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

def log_to_sheet(player, team, league, position, colour, comment, action="Updated"):
    try:
        sheet = init_sheet()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([player, team, league, position, colour, comment, action, now])
    except Exception as e:
        st.error(f"❌ Failed to log to Google Sheet: {e}")

# ============================================================
# DB Functions
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
    """Completely remove player from database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM favourites WHERE player=?", (player,))
    conn.commit()
    conn.close()

# ============================================================
# Load favourites
# ============================================================
show_hidden = st.checkbox("Show hidden players", value=False)
rows = get_favourites(show_hidden=show_hidden)

if not rows:
    st.info("No favourites saved yet.")
    st.stop()

df = pd.DataFrame(rows, columns=["Player", "Team", "League", "Position", "Colour", "Comment", "Visible", "Timestamp"])

# ============================================================
# Editable Table
# ============================================================
colour_options = ["", "🟢 Go", "🟡 Monitor", "🔴 No Further Interest", "🟣 Needs Checked"]

st.markdown("##### ✏️ Add A Comment or Colour")

edited_df = st.data_editor(
    df[["Player", "Team", "League", "Position", "Colour", "Comment", "Visible"]],
    column_config={
        "Colour": st.column_config.SelectboxColumn(
            "Colour",
            help="Set player status colour",
            options=colour_options,
            required=False,
            default=""
        ),
        "Comment": st.column_config.TextColumn(
            "Comment",
            help="Add notes or scouting comments about this player"
        ),
        "Visible": st.column_config.CheckboxColumn(
            "Visible",
            help="Uncheck to hide player (instead of deleting)"
        ),
    },
    hide_index=True,
    width="stretch",
)

# ============================================================
# Save and Sync Changes
# ============================================================
for idx, row in edited_df.iterrows():
    player = row["Player"]
    colour = row.get("Colour", "")
    comment = row.get("Comment", "")
    visible = int(row.get("Visible", True))

    prev = df.loc[df["Player"] == player].iloc[0]
    colour_changed = colour != prev["Colour"]
    comment_changed = comment != prev["Comment"]
    visible_changed = int(prev["Visible"]) != visible

    update_favourite(player, colour, comment, visible)

    if colour_changed or comment_changed or visible_changed:
        action = "Hidden" if visible == 0 else "Updated"
        log_to_sheet(player, row["Team"], row["League"], row["Position"], colour, comment, action)

# ============================================================
# Remove Buttons (Permanent delete)
# ============================================================
st.markdown("##### Permanently Remove a Favourite")

for _, row in df.iterrows():
    col1, col2 = st.columns([5, 1])
    with col1:
        st.write(f"**{row['Player']}** | {row['Team']} | {row['League']} | {row['Position']}")
    with col2:
        if st.button("🗑️ Remove", key=f"remove_{row['Player']}"):
            delete_favourite(row["Player"])
            log_to_sheet(row["Player"], row["Team"], row["League"], row["Position"], row["Colour"], row["Comment"], "Removed")
            st.success(f"Removed {row['Player']} from favourites.")
            st.rerun()

# ============================================================
# Summary
# ============================================================
visible_count = (edited_df["Visible"] == True).sum()
hidden_count = (edited_df["Visible"] == False).sum()
st.caption(f"Showing {visible_count} visible players ({hidden_count} hidden). Changes and removals are logged.")
