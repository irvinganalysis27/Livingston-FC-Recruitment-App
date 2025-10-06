# pages/7_Favourites.py

import streamlit as st
import sqlite3
from pathlib import Path
from datetime import datetime
import pandas as pd

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

# ============================================================
# Database setup (multi-comment tracking)
# ============================================================
DB_PATH = Path(__file__).parent / "favourites.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS favourites (
            player TEXT,
            team TEXT,
            league TEXT,
            position TEXT,
            colour TEXT DEFAULT 'Monitor',
            comment TEXT,
            user TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

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

def log_to_sheet(player, team, league, position, colour, comment, user, action):
    """Log every change to Google Sheets for audit trail."""
    try:
        sheet = init_sheet()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([player, team, league, position, colour, comment, user, action, now])
    except Exception as e:
        st.error(f"❌ Failed to log to Google Sheet: {e}")

# ============================================================
# Database helpers
# ============================================================
def get_latest_entries():
    """Fetch the most recent comment per player."""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT player, team, league, position, colour, comment, user, MAX(timestamp)
        FROM favourites
        GROUP BY player
        ORDER BY MAX(timestamp) DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_player_history(player):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM favourites WHERE player=? ORDER BY timestamp DESC",
        conn,
        params=(player,)
    )
    conn.close()
    return df

def add_comment(player, team, league, position, colour, comment, user):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO favourites (player, team, league, position, colour, comment, user)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (player, team, league, position, colour, comment, user))
    conn.commit()
    conn.close()
    log_to_sheet(player, team, league, position, colour, comment, user, "Added")

def remove_player(player):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM favourites WHERE player=?", (player,))
    conn.commit()
    conn.close()
    log_to_sheet(player, "-", "-", "-", "-", "-", "System", "Removed")

# ============================================================
# Colour tags
# ============================================================
COLOUR_OPTIONS = {
    "Go": "🟢 Go",
    "Monitor": "🟡 Monitor",
    "No Further Interest": "🔴 No Further Interest"
}

# ============================================================
# UI
# ============================================================
user_name = st.text_input("Your name (for tracking):", key="user_name")
if not user_name:
    st.warning("Please enter your name before making updates.")
    st.stop()

# Keep state for selected player (detail view)
if "selected_player" not in st.session_state:
    st.session_state.selected_player = None

# --- DETAIL VIEW ---
if st.session_state.selected_player:
    player = st.session_state.selected_player
    st.markdown(f"## 🧾 {player} – Comment History")

    df_history = get_player_history(player)
    if df_history.empty:
        st.info("No comments yet for this player.")
    else:
        st.dataframe(df_history[["timestamp", "user", "colour", "comment"]])

    st.markdown("### ➕ Add New Comment")
    new_comment = st.text_area("Comment", key="new_comment_text")
    new_colour = st.selectbox("Status", list(COLOUR_OPTIONS.values()), key="new_comment_colour")

    if st.button("💾 Submit Comment"):
        chosen_colour = [k for k, v in COLOUR_OPTIONS.items() if v == new_colour][0]
        info = df_history.iloc[0] if not df_history.empty else None
        team = info["team"] if info is not None else ""
        league = info["league"] if info is not None else ""
        position = info["position"] if info is not None else ""

        add_comment(player, team, league, position, chosen_colour, new_comment, user_name)
        st.success("Comment added successfully.")
        st.session_state.new_comment_text = ""
        st.rerun()

    if st.button("⬅️ Back to Favourites"):
        st.session_state.selected_player = None
        st.rerun()

# --- MAIN LIST VIEW ---
else:
    st.markdown("### Your Favourites")

    df_latest = get_latest_entries()
    if df_latest.empty:
        st.info("No favourites have been added yet.")
    else:
        for _, row in df_latest.iterrows():
            player, team, league, position, colour, comment, user, ts = row

            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

                with col1:
                    if st.button(f"👤 {player}", key=f"view_{player}"):
                        st.session_state.selected_player = player
                        st.rerun()
                    st.caption(f"{team} | {league} | {position}")

                with col2:
                    new_colour = st.selectbox(
                        "Status",
                        list(COLOUR_OPTIONS.values()),
                        index=list(COLOUR_OPTIONS.keys()).index(colour if colour in COLOUR_OPTIONS else "Monitor"),
                        key=f"colour_{player}"
                    )
                    chosen_colour = [k for k, v in COLOUR_OPTIONS.items() if v == new_colour][0]

                with col3:
                    new_comment = st.text_input(
                        f"Quick comment for {player}",
                        value=comment if comment else "",
                        key=f"comment_{player}"
                    )

                with col4:
                    if st.button("💾 Save", key=f"save_{player}"):
                        add_comment(player, team, league, position, chosen_colour, new_comment, user_name)
                        st.success(f"Saved update for {player}")
                        st.rerun()

                    if st.button("❌ Remove", key=f"remove_{player}"):
                        remove_player(player)
                        st.warning(f"{player} removed from favourites")
                        st.rerun()
