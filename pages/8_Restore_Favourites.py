import streamlit as st
import pandas as pd
import sqlite3

st.title("🧩 Restore Favourites from Google Sheet")

# --- Your Google Sheet (convert link to CSV export form) ---
sheet_csv_url = "https://docs.google.com/spreadsheets/d/16oweZkbqNst16U5lQshnYjwjWiuCmd8ClZfPyAtoE0o/export?format=csv&gid=0"

try:
    df = pd.read_csv(sheet_csv_url)
    st.success(f"✅ Loaded {len(df)} rows from Google Sheet")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"❌ Failed to load sheet: {e}")
    st.stop()

# --- Validate expected columns ---
expected = ["player", "rating", "comment"]
missing = [c for c in expected if c not in df.columns]
if missing:
    st.warning(f"Missing columns: {missing}. Please check the Google Sheet header.")
    st.stop()

# --- Prepare DataFrame ---
df = df.rename(columns={c: c.strip().lower() for c in df.columns})
df = df[["player", "rating", "comment"]].copy()
df["colour"] = ""       # blank for now
df["visible"] = 1       # visible = 1 (we’ll manually hide later if needed)
df["timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

# --- Save to SQLite ---
DB_PATH = "favourites.db"

conn = sqlite3.connect(DB_PATH)
df.to_sql("favourites", conn, if_exists="replace", index=False)
conn.close()

st.success("✅ Favourites successfully restored into favourites.db!")
st.info("You can now go back to the main app and your favourites should reappear.")
