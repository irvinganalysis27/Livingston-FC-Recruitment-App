import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

APP_DIR = Path(__file__).parent
ASSETS_DIR = APP_DIR / "assets"

def open_image(path: Path):
    try:
        return Image.open(path)
    except Exception:
        return None

# --- Basic password protection ---
PASSWORD = "Livi2025"

st.set_page_config(page_title="Livingston FC Recruitment App", layout="centered")

# ---------- Club Branding Row ----------
left, mid, right = st.columns([1, 6, 1])

logo_path = ASSETS_DIR / "Livingston_FC_club_badge_new.png"   # case-sensitive
logo = open_image(logo_path)

with left:
    if logo:
        st.image(logo, use_container_width=True)

with mid:
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1>Livingston FC Recruitment<br>App</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

with right:
    if logo:
        st.image(logo, use_container_width=True)

# Ask for password
pwd = st.text_input("Enter password:", type="password")
if pwd != PASSWORD:
    st.warning("Please enter the correct password to access the app.")
    st.stop()

# ========== Role groups shown in filters ==========
SIX_GROUPS = [
    "Goalkeeper",
    "Full Back",
    "Centre Back",
    "Number 6",
    "Number 8",
    "Winger",
    "Striker"
]

# ========== Position → group mapping for NEW PROVIDER labels ==========
def _clean_pos_token(tok: str) -> str:
    if pd.isna(tok):
        return ""
    t = str(tok).upper().strip()
    t = re.sub(r"[.\-_/]", " ", t)
    t = re.sub(r"\s+", "", t)
    return t

RAW_TO_SIX = {
    # GK
    "GOALKEEPER": "Goalkeeper",

    # Full backs & wing backs
    "RIGHTBACK": "Full Back",
    "LEFTBACK": "Full Back",
    "RIGHTWINGBACK": "Full Back",
    "LEFTWINGBACK": "Full Back",

    # Centre backs
    "RIGHTCENTREBACK": "Centre Back",
    "LEFTCENTREBACK": "Centre Back",
    "CENTREBACK": "Centre Back",

    # Centre mid (generic) → we’ll duplicate into 6 & 8 later
    "CENTREMIDFIELDER": "Centre Midfield",
    "RIGHTCENTREMIDFIELDER": "Centre Midfield",
    "LEFTCENTREMIDFIELDER": "Centre Midfield",

    # Defensive mids → 6
    "DEFENSIVEMIDFIELDER": "Number 6",
    "RIGHTDEFENSIVEMIDFIELDER": "Number 6",
    "LEFTDEFENSIVEMIDFIELDER": "Number 6",

    # Attacking mids / 10 → 8
    "CENTREATTACKINGMIDFIELDER": "Number 8",
    "ATTACKINGMIDFIELDER": "Number 8",
    "SECONDSTRIKER": "Number 8",
    "10": "Number 8",

    # Wingers / wide mids
    "RIGHTWING": "Winger",
    "LEFTWING": "Winger",
    "RIGHTMIDFIELDER": "Winger",
    "LEFTMIDFIELDER": "Winger",

    # Strikers
    "CENTREFORWARD": "Striker",
    "RIGHTCENTREFORWARD": "Striker",
    "LEFTCENTREFORWARD": "Striker",
}

def parse_first_position(cell) -> str:
    # We’ll use PRIMARY position only for mapping; UI will show both
    if pd.isna(cell):
        return ""
    return _clean_pos_token(str(cell))

def map_first_position_to_group(primary_pos_cell) -> str:
    tok = parse_first_position(primary_pos_cell)
    return RAW_TO_SIX.get(tok, "Winger")  # safe default

# ========== Default template mapping ==========
DEFAULT_TEMPLATE = {
    "Goalkeeper": "Goalkeeper",
    "Full Back": "Full Back",
    "Centre Back": "Centre Back",
    "Number 6": "Number 6",
    "Number 8": "Number 8",
    "Winger": "Winger",
    "Striker": "Striker"
}

# ========== Radar metric sets (NEW PROVIDER NAMES ONLY) ==========
position_metrics = {
    "Goalkeeper": {
        "metrics": [
            # Keepers (retain only if your new file has them; otherwise they'll show as 0s)
            "Goals Conceded", "PSxG Faced", "GSAA", "Save%", "xSv%", "Shot Stopping%",
            "Shots Faced", "Shots Faced OT%",
            "Pass into Danger%", "Pass into Pressure%",
            "Positive Outcome%", "Goalkeeper OBV"
        ],
        "groups": {
            "Goals Conceded": "Goalkeeping",
            "PSxG Faced": "Goalkeeping",
            "GSAA": "Goalkeeping",
            "Save%": "Goalkeeping",
            "xSv%": "Goalkeeping",
            "Shot Stopping%": "Goalkeeping",
            "Shots Faced": "Goalkeeping",
            "Shots Faced OT%": "Goalkeeping",
            "Pass into Danger%": "Possession",
            "Pass into Pressure%": "Possession",
            "Positive Outcome%": "Goalkeeping",
            "Goalkeeper OBV": "Goalkeeping",
        }
    },

    "Centre Back": {
        "metrics": [
            "PAdj Interceptions", "PAdj Tackles", "Tack/DP%",
            "Defensive Actions", "Ball Recoveries",
            "Aerial Win%", "Aerial Wins",
            "Passing%", "Pass OBV",
            "UPr. Long Balls",
            "Pressures", "PAdj Pressures"
        ],
        "groups": {
            "PAdj Interceptions": "Defensive",
            "PAdj Tackles": "Defensive",
            "Tack/DP%": "Defensive",
            "Defensive Actions": "Defensive",
            "Ball Recoveries": "Defensive",
            "Aerial Win%": "Defensive",
            "Aerial Wins": "Defensive",
            "Passing%": "Possession",
            "Pass OBV": "Possession",
            "UPr. Long Balls": "Possession",
            "Pressures": "Off The Ball",
            "PAdj Pressures": "Off The Ball",
        }
    },

    "Full Back": {
        "metrics": [
            "Pass OBV", "Passing%", "OP Passes Into Box",
            "Deep Progressions", "Deep Completions",
            "Successful Dribbles", "Dribbles",
            "Turnovers",
            "Defensive Actions",
            "Aerial Win%", "Aerial Wins",
            "Pressures", "PAdj Pressures",
            "Tack/DP%"
        ],
        "groups": {
            "Pass OBV": "Possession",
            "Passing%": "Possession",
            "OP Passes Into Box": "Possession",
            "Deep Progressions": "Possession",
            "Deep Completions": "Possession",
            "Successful Dribbles": "Possession",
            "Dribbles": "Possession",
            "Turnovers": "Possession",
            "Defensive Actions": "Defensive",
            "Aerial Win%": "Defensive",
            "Aerial Wins": "Defensive",
            "Pressures": "Off The Ball",
            "PAdj Pressures": "Off The Ball",
            "Tack/DP%":
