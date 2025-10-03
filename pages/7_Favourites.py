import streamlit as st
import sqlite3
from pathlib import Path
from auth import check_password
from branding import show_branding

# Protect page
if not check_password():
    st.stop()

show_branding()
st.title("⭐ Favourite Players")

# Path to DB
DB_PATH = Path(__file__).parent / "favourites.db"

def get_favourites():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT player, team, league, position, timestamp FROM favourites ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return rows

def remove_favourite(player):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM favourites WHERE player=?", (player,))
    conn.commit()
    conn.close()

# Fetch and display favourites
favs = get_favourites()

if favs:
    st.markdown("### Your Favourites")

    for player, team, league, position, ts in favs:
        col1, col2 = st.columns([5,1])
        with col1:
            st.write(f"**{player}** | {team} | {league} | {position}")
        with col2:
            if st.button("❌ Remove", key=f"remove_{player}"):
                remove_favourite(player)
                st.success(f"Removed {player} from favourites")
                st.experimental_rerun()

else:
    st.info("No favourites have been added yet.")
