from pathlib import Path
import sqlite3
import pandas as pd

DB_PATH = Path("/mount/src/livingston-fc-recruitment-app/pages/favourites.db")

sheet_csv_url = "https://docs.google.com/spreadsheets/d/16oweZkbqNst16U5lQshnYjwjWiuCmd8ClZfPyAtoE0o/export?format=csv&gid=0"
df = pd.read_csv(sheet_csv_url)

# clean and map the columns
df = df.rename(columns={c.strip().lower(): c for c in df.columns})
df = df[["player", "rating", "comment"]].copy()
df["colour"] = df["rating"].fillna("")
df["visible"] = 1
df["timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

# write to database
conn = sqlite3.connect(DB_PATH)
df.to_sql("favourites", conn, if_exists="replace", index=False)
conn.close()

print(f"✅ Restored {len(df)} favourites to {DB_PATH}")
