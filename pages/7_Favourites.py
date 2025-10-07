import streamlit as st
import sqlite3
from pathlib import Path
from datetime import datetime
from auth import check_password
from branding import show_branding

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

    # Base table
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

    # Migration for missing columns
    existing_cols = [row[1] for row in c.execute("PRAGMA table_info(favourites)").fetchall()]
    if "colour" not in existing_cols:
        c.execute("ALTER TABLE favourites ADD COLUMN colour TEXT DEFAULT ''")
    if "comment" not in existing_cols:
        c.execute("ALTER TABLE favourites ADD COLUMN comment TEXT DEFAULT ''")
    if "visible" not in existing_cols:
        c.execute("ALTER TABLE favourites ADD COLUMN visible INTEGER DEFAULT 1")

    conn.commit()
    conn.close()

init_db()

# ============================================================
# Helper DB functions
# ============================================================
def get_favourites(show_hidden=False):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if show_hidden:
        c.execute("SELECT player, team, league, position, colour, comment, visible, timestamp FROM favourites ORDER BY timestamp DESC")
    else:
        c.execute("SELECT player, team, league, position, colour, comment, visible, timestamp FROM favourites WHERE visible=1 ORDER BY timestamp DESC")
    rows = c.fetchall()
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

def restore_favourite(player):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE favourites SET visible=1 WHERE player=?", (player,))
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

# Build DataFrame for display
import pandas as pd
df = pd.DataFrame(rows, columns=["Player", "Team", "League", "Position", "Colour", "Comment", "Visible", "Timestamp"])

# ============================================================
# Table Configuration
# ============================================================
colour_options = ["", "🟢 Green", "🟡 Yellow", "🔴 Red", "🟣 Purple"]

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
# Save changes automatically
# ============================================================
for _, row in edited_df.iterrows():
    player = row["Player"]
    update_favourite(
        player,
        row.get("Colour", ""),
        row.get("Comment", ""),
        int(row.get("Visible", True))
    )

# ============================================================
# Optional summary
# ============================================================
visible_count = (edited_df["Visible"] == True).sum()
hidden_count = (edited_df["Visible"] == False).sum()

st.caption(f"Showing {visible_count} visible players ({hidden_count} hidden). All changes are auto-saved.")
