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

# Fetch and display favourites
favs = get_favourites()

if favs:
    st.markdown("### Your Favourites")
    st.table(favs)   # or st.dataframe if you want sorting/filtering
else:
    st.info("No favourites have been added yet.")
