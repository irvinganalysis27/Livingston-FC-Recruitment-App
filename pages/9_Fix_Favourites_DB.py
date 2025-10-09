import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "favourites.db"
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# --- Create (or replace) favourites table with correct schema ---
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

# --- Ensure all columns exist ---
existing_cols = [r[1] for r in c.execute("PRAGMA table_info(favourites)").fetchall()]
required_cols = {
    "colour": "TEXT DEFAULT ''",
    "comment": "TEXT DEFAULT ''",
    "visible": "INTEGER DEFAULT 1",
    "timestamp": "DATETIME DEFAULT CURRENT_TIMESTAMP"
}

for col, dtype in required_cols.items():
    if col not in existing_cols:
        print(f"Adding missing column: {col}")
        c.execute(f"ALTER TABLE favourites ADD COLUMN {col} {dtype}")

conn.commit()
conn.close()

print("✅ favourites.db schema fixed and verified")
