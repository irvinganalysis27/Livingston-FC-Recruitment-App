import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent / "favourites.db"

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM favourites", conn)

print("Before cleanup:", len(df))

# --- Remove duplicates based on player name, keeping latest timestamp ---
df = df.sort_values("timestamp").drop_duplicates(subset=["player"], keep="last")

df.to_sql("favourites", conn, if_exists="replace", index=False)
conn.close()

print("After cleanup:", len(df))
print("✅ Duplicates removed successfully.")
