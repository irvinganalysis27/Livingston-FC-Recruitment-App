from pathlib import Path
import sqlite3
import pandas as pd
import streamlit as st  # so we can print messages in the app

DB_PATH = Path("/mount/src/livingston-fc-recruitment-app/pages/favourites.db")

sheet_csv_url = "https://docs.google.com/spreadsheets/d/16oweZkbqNst16U5lQshnYjwjWiuCmd8ClZfPyAtoE0o/export?format=csv&gid=0"
df = pd.read_csv(sheet_csv_url)

st.write("✅ Columns loaded from Google Sheet:", list(df.columns))

# --- Standardise column names ---
df.columns = df.columns.str.strip().str.lower()

# --- Select & rename the relevant ones ---
df = df.rename(columns={
    "player": "player",
    "team": "team",
    "league": "league",
    "position": "position",
    "rating": "colour",     # map Rating -> colour
    "comment": "comment",
})

# --- Keep only the required columns for DB ---
keep_cols = ["player", "team", "league", "position", "colour", "comment"]
df = df[keep_cols].copy()

# --- Mark everything as visible ---
df["visible"] = 1
df["timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

# --- Save to database ---
conn = sqlite3.connect(DB_PATH)
df.to_sql("favourites", conn, if_exists="replace", index=False)
conn.close()

st.success(f"✅ Restored {len(df)} favourites to {DB_PATH}")
