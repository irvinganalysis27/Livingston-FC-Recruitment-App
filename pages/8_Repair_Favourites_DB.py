import streamlit as st
import sqlite3
from pathlib import Path

st.set_page_config(page_title="Repair Favourites DB", layout="centered")

st.title("🛠️ Repair Favourites Database")

DB_PATH = Path(__file__).parent / "favourites.db"

if st.button("Run Repair"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Check what columns exist
    c.execute("PRAGMA table_info(favourites)")
    cols = [r[1] for r in c.fetchall()]
    st.write("Existing columns before repair:", cols)

    # Add missing columns if needed
    added = []
    if "timestamp" not in cols:
        c.execute("ALTER TABLE favourites ADD COLUMN timestamp DATETIME DEFAULT CURRENT_TIMESTAMP")
        added.append("timestamp")
    if "visible" not in cols:
        c.execute("ALTER TABLE favourites ADD COLUMN visible INTEGER DEFAULT 1")
        added.append("visible")
    if "colour" not in cols:
        c.execute("ALTER TABLE favourites ADD COLUMN colour TEXT DEFAULT ''")
        added.append("colour")
    if "comment" not in cols:
        c.execute("ALTER TABLE favourites ADD COLUMN comment TEXT DEFAULT ''")
        added.append("comment")

    conn.commit()
    conn.close()

    if added:
        st.success(f"✅ Added missing columns: {', '.join(added)}")
    else:
        st.info("✅ All columns already existed — no changes needed.")

    st.write("Database repair completed successfully.")
else:
    st.warning("Click the button above to check and repair the database schema.")
