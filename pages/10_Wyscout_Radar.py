import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from PIL import Image
from datetime import datetime, timezone

# ========= PAGE CONFIG =========
st.set_page_config(page_title="Livingston FC Recruitment App", layout="centered")

# ========= AUTH / BRANDING =========
from auth import check_password
from branding import show_branding

# --- Password gate ---
if not check_password():
    st.stop()

# --- Club branding header ---
show_branding()
st.title("Wyscout Radar")

# ========= PATHS =========
APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent
ASSETS_DIR = ROOT_DIR / "assets"

def open_image(path: Path):
    """Safe image loader."""
    try:
        return Image.open(path)
    except Exception:
        return None

# ========= USER INSTRUCTIONS =========
st.markdown("""
### 📋 How to Download Data from Wyscout

1. **Open Wyscout**
2. Click the **menu (top centre)**.
3. Choose **Advanced Search**.
4. Under *Competition*, select your **League**.
5. Choose the **most recent season**.
6. At the top-right, set **Display → All**.
7. Click **Download → Excel**.
8. If there are too many players (over ~500), use the **Position filters (left side)** to narrow down the list.

Once downloaded, upload that Excel file below to generate the radar charts and ranking table.
""")
st.divider()

# ================== Color helpers for tercile gradient bars ==================
def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _rgb_to_hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"

def _lerp(c1, c2, t):
    r1,g1,b1 = _hex_to_rgb(c1); r2,g2,b2 = _hex_to_rgb(c2)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return _rgb_to_hex(r, g, b)

# band endpoints (light -> deep) for each tercile
BAND_PALETTES = {
    "low":  ("#fee2e2", "#ef4444"),  # red
    "mid":  ("#fff7ed", "#f59e0b"),  # orange
    "high": ("#dcfce7", "#16a34a"),  # green
}

def percentile_to_color(p):
    """Map 0..100 percentile to a gradient within terciles (low/mid/high)."""
    p = 0 if p is None else float(p)
    if p < 33.3334:
        lo, hi = BAND_PALETTES["low"]
        t = p / 33.3334
        return _lerp(lo, hi, t)
    elif p < 66.6667:
        lo, hi = BAND_PALETTES["mid"]
        t = (p - 33.3334) / 33.3333
        return _lerp(lo, hi, t)
    else:
        lo, hi = BAND_PALETTES["high"]
        t = (p - 66.6667) / 33.3333
        return _lerp(lo, hi, t)

# background wedge opacity + outside genre label radius
GENRE_BG_ALPHA   = 0.14      # a touch stronger so averages read well
GENRE_LABEL_R    = 118       # outside the 0..100 ring
SECTION_GAP_FRAC = 0.15      # fraction of one bar's step trimmed from EACH side of a section

# ================== 6-position mapping ==================
SIX_GROUPS = [
    "Goalkeeper", "Wide Defender", "Central Defender",
    "Central Midfielder", "Wide Midfielder", "Central Forward"
]

RAW_TO_SIX = {
    "GK": "Goalkeeper", "GKP": "Goalkeeper", "GOALKEEPER": "Goalkeeper",
    "RB": "Wide Defender", "LB": "Wide Defender",
    "RWB": "Wide Defender", "LWB": "Wide Defender", "RFB": "Wide Defender", "LFB": "Wide Defender",
    "CB": "Central Defender", "RCB": "Central Defender", "LCB": "Central Defender",
    "CBR": "Central Defender", "CBL": "Central Defender", "SW": "Central Defender",
    "CMF": "Central Midfielder", "CM": "Central Midfielder", "RCMF": "Central Midfielder", "RCM": "Central Midfielder",
    "LCMF": "Central Midfielder", "LCM": "Central Midfielder", "DMF": "Central Midfielder",
    "DM": "Central Midfielder", "CDM": "Central Midfielder", "RDMF": "Central Midfielder", "RDM": "Central Midfielder",
    "LDMF": "Central Midfielder", "LDM": "Central Midfielder", "AMF": "Central Midfielder",
    "AM": "Central Midfielder", "CAM": "Central Midfielder", "SS": "Central Midfielder", "10": "Central Midfielder",
    "LWF": "Wide Midfielder", "RWF": "Wide Midfielder", "RW": "Wide Midfielder", "LW": "Wide Midfielder",
    "LAMF": "Wide Midfielder", "RAMF": "Wide Midfielder", "RM": "Wide Midfielder", "LM": "Wide Midfielder",
    "WF": "Wide Midfielder", "RWG": "Wide Midfielder", "LWG": "Wide Midfielder", "W": "Wide Midfielder",
    "CF": "Central Forward", "ST": "Central Forward", "9": "Central Forward",
    "FW": "Central Forward", "STK": "Central Forward", "CFW": "Central Forward"
}

def _clean_pos_token(tok: str) -> str:
    if pd.isna(tok):
        return ""
    t = str(tok).upper()
    t = t.replace(".", "").replace("-", "").replace(" ", "")
    return t

def parse_first_position(cell) -> str:
    if pd.isna(cell):
        return ""
    first = re.split(r"[,/]", str(cell))[0].strip()
    return _clean_pos_token(first)

def map_first_position_to_group(cell) -> str:
    tok = parse_first_position(cell)
    return RAW_TO_SIX.get(tok, "Wide Midfielder")  # safe default

# ================== Default template mapping ==================
DEFAULT_TEMPLATE = {
    "Goalkeeper": "Goalkeeper",
    "Wide Defender": "Wide Defender, Full Back",
    "Central Defender": "Central Defender, All Round",
    "Central Midfielder": "Central Midfielder, All Round CM",
    "Wide Midfielder": "Wide Midfielder, Touchline Winger",
    "Central Forward": "Striker, All Round CF"
}

# ========== Metric sets ==========
position_metrics = {
    # ================== GOALKEEPER ==================
    "Goalkeeper": {
        "metrics": [
            "Clean sheets per 90", "Conceded goals per 90", "Prevented goals per 90",
            "Save rate, %", "Shots against per 90", "Aerial duels per 90", "Exits per 90",
            "Passes per 90", "Accurate passes, %", "Short / medium passes per 90",
            "Accurate short / medium passes, %", "Long passes per 90", "Accurate long passes, %"
        ],
        "groups": {
            "Clean sheets per 90": "Goalkeeping",
            "Conceded goals per 90": "Goalkeeping",
            "Prevented goals per 90": "Goalkeeping",
            "Save rate, %": "Goalkeeping",
            "Shots against per 90": "Goalkeeping",
            "Aerial duels per 90": "Goalkeeping",
            "Exits per 90": "Goalkeeping",
            "Passes per 90": "Possession",
            "Accurate passes, %": "Possession",
            "Short / medium passes per 90": "Possession",
            "Accurate short / medium passes, %": "Possession",
            "Long passes per 90": "Possession",
            "Accurate long passes, %": "Possession"
        }
    },

    # ================== CENTRAL DEFENDERS ==================
    "Central Defender, Ball Winning": {
        "metrics": [
            "Defensive duels per 90", "Defensive duels won, %",
            "Aerial duels per 90", "Aerial duels won, %",
            "Shots blocked per 90", "PAdj Interceptions",
            "Head goals per 90", "Successful dribbles, %", "Accurate passes, %"
        ],
        "groups": {
            "Defensive duels per 90": "Defensive",
            "Defensive duels won, %": "Defensive",
            "Aerial duels per 90": "Defensive",
            "Aerial duels won, %": "Defensive",
            "Shots blocked per 90": "Defensive",
            "PAdj Interceptions": "Defensive",
            "Head goals per 90": "Attacking",
            "Successful dribbles, %": "Possession",
            "Accurate passes, %": "Possession"
        }
    },
    "Central Defender, Ball Playing": {
        "metrics": [
            "Defensive duels per 90", "Defensive duels won, %",
            "Shots blocked per 90", "PAdj Interceptions",
            "Forward passes per 90", "Accurate forward passes, %",
            "Passes to final third per 90", "Accurate passes to final third, %",
            "Accurate passes, %", "Dribbles per 90", "Successful dribbles, %"
        ],
        "groups": {
            "Defensive duels per 90": "Defensive",
            "Defensive duels won, %": "Defensive",
            "Shots blocked per 90": "Defensive",
            "PAdj Interceptions": "Defensive",
            "Forward passes per 90": "Possession",
            "Accurate forward passes, %": "Possession",
            "Passes to final third per 90": "Possession",
            "Accurate passes to final third, %": "Possession",
            "Accurate passes, %": "Possession",
            "Dribbles per 90": "Possession",
            "Successful dribbles, %": "Possession"
        }
    },
    "Central Defender, All Round": {
        "metrics": [
            "Defensive duels per 90", "Defensive duels won, %",
            "Aerial duels per 90", "Aerial duels won, %",
            "Shots blocked per 90", "PAdj Interceptions",
            "Accurate passes, %", "Forward passes per 90", "Accurate forward passes, %",
            "Passes to final third per 90", "Accurate passes to final third, %",
            "Dribbles per 90", "Successful dribbles, %"
        ],
        "groups": {
            "Defensive duels per 90": "Defensive",
            "Defensive duels won, %": "Defensive",
            "Aerial duels per 90": "Defensive",
            "Aerial duels won, %": "Defensive",
            "Shots blocked per 90": "Defensive",
            "PAdj Interceptions": "Defensive",
            "Accurate passes, %": "Possession",
            "Forward passes per 90": "Possession",
            "Accurate forward passes, %": "Possession",
            "Passes to final third per 90": "Possession",
            "Accurate passes to final third, %": "Possession",
            "Dribbles per 90": "Possession",
            "Successful dribbles, %": "Possession"
        }
    },

    # ================== WIDE DEFENDERS ==================
    "Wide Defender, Full Back": {
        "metrics": [
            "Successful defensive actions per 90", "Defensive duels per 90", "Defensive duels won, %",
            "PAdj Interceptions", "Crosses per 90", "Accurate crosses, %",
            "Passes to final third per 90", "Accurate passes to final third, %",
            "Dribbles per 90", "Successful dribbles, %",
            "xA per 90", "Assists per 90"
        ],
        "groups": {
            "Successful defensive actions per 90": "Defensive",
            "Defensive duels per 90": "Defensive",
            "Defensive duels won, %": "Defensive",
            "PAdj Interceptions": "Defensive",
            "Crosses per 90": "Possession",
            "Accurate crosses, %": "Possession",
            "Passes to final third per 90": "Possession",
            "Accurate passes to final third, %": "Possession",
            "Dribbles per 90": "Possession",
            "Successful dribbles, %": "Possession",
            "xA per 90": "Attacking",
            "Assists per 90": "Attacking"
        }
    },
    "Wide Defender, Wing Back": {
        "metrics": [
            "Successful defensive actions per 90", "Defensive duels per 90", "Defensive duels won, %",
            "Dribbles per 90", "Successful dribbles, %", "Offensive duels per 90", "Offensive duels won, %",
            "Crosses per 90", "Accurate crosses, %", "Passes to final third per 90",
            "xA per 90", "Assists per 90", "Shot assists per 90"
        ],
        "groups": {
            "Successful defensive actions per 90": "Defensive",
            "Defensive duels per 90": "Defensive",
            "Defensive duels won, %": "Defensive",
            "Dribbles per 90": "Possession",
            "Successful dribbles, %": "Possession",
            "Offensive duels per 90": "Possession",
            "Offensive duels won, %": "Possession",
            "Crosses per 90": "Possession",
            "Accurate crosses, %": "Possession",
            "Passes to final third per 90": "Possession",
            "xA per 90": "Attacking",
            "Assists per 90": "Attacking",
            "Shot assists per 90": "Attacking"
        }
    },
    "Wide Defender, Inverted": {
        "metrics": [
            "Successful defensive actions per 90", "Defensive duels per 90", "Defensive duels won, %",
            "PAdj Interceptions", "Forward passes per 90", "Accurate forward passes, %",
            "Through passes per 90", "Accurate through passes, %",
            "Passes to final third per 90", "Accurate passes to final third, %",
            "xA per 90", "Assists per 90"
        ],
        "groups": {
            "Successful defensive actions per 90": "Defensive",
            "Defensive duels per 90": "Defensive",
            "Defensive duels won, %": "Defensive",
            "PAdj Interceptions": "Defensive",
            "Forward passes per 90": "Possession",
            "Accurate forward passes, %": "Possession",
            "Through passes per 90": "Possession",
            "Accurate through passes, %": "Possession",
            "Passes to final third per 90": "Possession",
            "Accurate passes to final third, %": "Possession",
            "xA per 90": "Attacking",
            "Assists per 90": "Attacking"
        }
    },

    # ================== CENTRAL MIDFIELDERS ==================
    "Central Midfielder, Creative": {
        "metrics": [
            "Non-penalty goals per 90", "xG per 90", "Goal conversion, %",
            "Assists per 90", "xA per 90", "Shots per 90", "Shots on target, %",
            "Forward passes per 90", "Accurate forward passes, %",
            "Through passes per 90", "Accurate through passes, %",
            "Dribbles per 90", "Successful dribbles, %"
        ],
        "groups": {
            "Non-penalty goals per 90": "Attacking",
            "xG per 90": "Attacking",
            "Goal conversion, %": "Attacking",
            "Assists per 90": "Attacking",
            "xA per 90": "Attacking",
            "Shots per 90": "Attacking",
            "Shots on target, %": "Attacking",
            "Forward passes per 90": "Possession",
            "Accurate forward passes, %": "Possession",
            "Through passes per 90": "Possession",
            "Accurate through passes, %": "Possession",
            "Dribbles per 90": "Possession",
            "Successful dribbles, %": "Possession"
        }
    },
    "Central Midfielder, Defensive": {
        "metrics": [
            "Successful defensive actions per 90", "Defensive duels per 90", "Defensive duels won, %",
            "Aerial duels per 90", "Aerial duels won, %", "PAdj Interceptions",
            "Successful dribbles, %", "Offensive duels per 90", "Offensive duels won, %",
            "Accurate passes, %", "Forward passes per 90", "Accurate forward passes, %",
            "Passes to final third per 90", "Accurate passes to final third, %"
        ],
        "groups": {
            "Successful defensive actions per 90": "Defensive",
            "Defensive duels per 90": "Defensive",
            "Defensive duels won, %": "Defensive",
            "Aerial duels per 90": "Defensive",
            "Aerial duels won, %": "Defensive",
            "PAdj Interceptions": "Defensive",
            "Successful dribbles, %": "Possession",
            "Offensive duels per 90": "Possession",
            "Offensive duels won, %": "Possession",
            "Accurate passes, %": "Possession",
            "Forward passes per 90": "Possession",
            "Accurate forward passes, %": "Possession",
            "Passes to final third per 90": "Possession",
            "Accurate passes to final third, %": "Possession"
        }
    },
    "Central Midfielder, All Round CM": {
        "metrics": [
            "Non-penalty goals per 90", "xG per 90", "Goal conversion, %",
            "Assists per 90", "xA per 90", "Shots per 90", "Shots on target, %",
            "Forward passes per 90", "Accurate forward passes, %",
            "Passes to final third per 90", "Accurate passes to final third, %",
            "Dribbles per 90", "Successful dribbles, %",
            "Successful defensive actions per 90", "Defensive duels per 90",
            "Defensive duels won, %", "PAdj Interceptions"
        ],
        "groups": {
            "Non-penalty goals per 90": "Attacking",
            "xG per 90": "Attacking",
            "Goal conversion, %": "Attacking",
            "Assists per 90": "Attacking",
            "xA per 90": "Attacking",
            "Shots per 90": "Attacking",
            "Shots on target, %": "Attacking",
            "Forward passes per 90": "Possession",
            "Accurate forward passes, %": "Possession",
            "Passes to final third per 90": "Possession",
            "Accurate passes to final third, %": "Possession",
            "Dribbles per 90": "Possession",
            "Successful dribbles, %": "Possession",
            "Successful defensive actions per 90": "Defensive",
            "Defensive duels per 90": "Defensive",
            "Defensive duels won, %": "Defensive",
            "PAdj Interceptions": "Defensive"
        }
    },

    # ================== WIDE MIDFIELDERS ==================
    "Wide Midfielder, Touchline Winger": {
        "metrics": [
            "Non-penalty goals per 90", "xG per 90", "Assists per 90", "xA per 90",
            "Crosses per 90", "Accurate crosses, %", "Dribbles per 90", "Successful dribbles, %",
            "Fouls suffered per 90", "Shot assists per 90",
            "Passes to penalty area per 90", "Accurate passes to penalty area, %"
        ],
        "groups": {
            "Non-penalty goals per 90": "Attacking",
            "xG per 90": "Attacking",
            "Assists per 90": "Attacking",
            "xA per 90": "Attacking",
            "Crosses per 90": "Possession",
            "Accurate crosses, %": "Possession",
            "Dribbles per 90": "Possession",
            "Successful dribbles, %": "Possession",
            "Fouls suffered per 90": "Possession",
            "Shot assists per 90": "Possession",
            "Passes to penalty area per 90": "Possession",
            "Accurate passes to penalty area, %": "Possession"
        }
    },
    "Wide Midfielder, Inverted Winger": {
        "metrics": [
            "Non-penalty goals per 90", "xG per 90", "Shots per 90", "Shots on target, %",
            "Goal conversion, %", "Assists per 90", "xA per 90",
            "Dribbles per 90", "Successful dribbles, %",
            "Fouls suffered per 90", "Shot assists per 90",
            "Passes to penalty area per 90", "Accurate passes to penalty area, %",
            "Deep completions per 90"
        ],
        "groups": {
            "Non-penalty goals per 90": "Attacking",
            "xG per 90": "Attacking",
            "Shots per 90": "Attacking",
            "Shots on target, %": "Attacking",
            "Goal conversion, %": "Attacking",
            "Assists per 90": "Attacking",
            "xA per 90": "Attacking",
            "Dribbles per 90": "Possession",
            "Successful dribbles, %": "Possession",
            "Fouls suffered per 90": "Possession",
            "Shot assists per 90": "Possession",
            "Passes to penalty area per 90": "Possession",
            "Accurate passes to penalty area, %": "Possession",
            "Deep completions per 90": "Possession"
        }
    },
    "Wide Midfielder, Defensive Wide Midfielder": {
        "metrics": [
            "Non-penalty goals per 90", "xG per 90", "Shots per 90", "Shots on target, %",
            "Assists per 90", "xA per 90",
            "Crosses per 90", "Accurate crosses, %", "Dribbles per 90", "Successful dribbles, %",
            "Fouls suffered per 90", "Shot assists per 90",
            "Successful defensive actions per 90", "Defensive duels won, %", "PAdj Interceptions"
        ],
        "groups": {
            "Non-penalty goals per 90": "Attacking",
            "xG per 90": "Attacking",
            "Shots per 90": "Attacking",
            "Shots on target, %": "Attacking",
            "Assists per 90": "Attacking",
            "xA per 90": "Attacking",
            "Crosses per 90": "Possession",
            "Accurate crosses, %": "Possession",
            "Dribbles per 90": "Possession",
            "Successful dribbles, %": "Possession",
            "Fouls suffered per 90": "Possession",
            "Shot assists per 90": "Possession",
            "Successful defensive actions per 90": "Defensive",
            "Defensive duels won, %": "Defensive",
            "PAdj Interceptions": "Defensive"
        }
    },

    # ================== STRIKERS ==================
    "Striker, Number 10": {
        "metrics": [
            "Successful defensive actions per 90",
            "Non-penalty goals per 90", "xG per 90", "Shots per 90", "Shots on target, %",
            "Goal conversion, %", "Assists per 90", "xA per 90", "Shot assists per 90",
            "Forward passes per 90", "Accurate forward passes, %",
            "Passes to final third per 90", "Accurate passes to final third, %",
            "Through passes per 90", "Accurate through passes, %"
        ],
        "groups": {
            "Successful defensive actions per 90": "Off The Ball",
            "Non-penalty goals per 90": "Attacking",
            "xG per 90": "Attacking",
            "Shots per 90": "Attacking",
            "Shots on target, %": "Attacking",
            "Goal conversion, %": "Attacking",
            "Assists per 90": "Attacking",
            "xA per 90": "Attacking",
            "Shot assists per 90": "Attacking",
            "Forward passes per 90": "Possession",
            "Accurate forward passes, %": "Possession",
            "Passes to final third per 90": "Possession",
            "Accurate passes to final third, %": "Possession",
            "Through passes per 90": "Possession",
            "Accurate through passes, %": "Possession"
        }
    },
    "Striker, Target Man": {
        "metrics": [
            "Aerial duels per 90", "Aerial duels won, %",
            "Non-penalty goals per 90", "xG per 90", "Shots per 90", "Shots on target, %",
            "Goal conversion, %", "Head goals per 90", "Assists per 90", "xA per 90", "Shot assists per 90",
            "Offensive duels per 90", "Offensive duels won, %",
            "Passes to penalty area per 90", "Accurate passes to penalty area, %"
        ],
        "groups": {
            "Aerial duels per 90": "Off The Ball",
            "Aerial duels won, %": "Off The Ball",
            "Non-penalty goals per 90": "Attacking",
            "xG per 90": "Attacking",
            "Shots per 90": "Attacking",
            "Shots on target, %": "Attacking",
            "Goal conversion, %": "Attacking",
            "Head goals per 90": "Attacking",
            "Assists per 90": "Attacking",
            "xA per 90": "Attacking",
            "Shot assists per 90": "Attacking",
            "Offensive duels per 90": "Possession",
            "Offensive duels won, %": "Possession",
            "Passes to penalty area per 90": "Possession",
            "Accurate passes to penalty area, %": "Possession"
        }
    },
    "Striker, Penalty Box Striker": {
        "metrics": [
            "Non-penalty goals per 90", "xG per 90", "Shots per 90", "Shots on target, %",
            "Goal conversion, %", "Shot assists per 90", "Touches in penalty area per 90",
            "Offensive duels per 90", "Offensive duels won, %",
            "Dribbles per 90", "Successful dribbles, %"
        ],
        "groups": {
            "Non-penalty goals per 90": "Attacking",
            "xG per 90": "Attacking",
            "Shots per 90": "Attacking",
            "Shots on target, %": "Attacking",
            "Goal conversion, %": "Attacking",
            "Shot assists per 90": "Attacking",
            "Touches in penalty area per 90": "Attacking",
            "Offensive duels per 90": "Possession",
            "Offensive duels won, %": "Possession",
            "Dribbles per 90": "Possession",
            "Successful dribbles, %": "Possession"
        }
    },
    "Striker, All Round CF": {
        "metrics": [
            "Successful defensive actions per 90", "Aerial duels per 90", "Aerial duels won, %",
            "Non-penalty goals per 90", "xG per 90", "Shots per 90", "Shots on target, %",
            "Goal conversion, %", "Assists per 90", "xA per 90", "Shot assists per 90",
            "Offensive duels per 90", "Offensive duels won, %"
        ],
        "groups": {
            "Successful defensive actions per 90": "Off The Ball",
            "Aerial duels per 90": "Off The Ball",
            "Aerial duels won, %": "Off The Ball",
            "Non-penalty goals per 90": "Attacking",
            "xG per 90": "Attacking",
            "Shots per 90": "Attacking",
            "Shots on target, %": "Attacking",
            "Goal conversion, %": "Attacking",
            "Assists per 90": "Attacking",
            "xA per 90": "Attacking",
            "Shot assists per 90": "Attacking",
            "Offensive duels per 90": "Possession",
            "Offensive duels won, %": "Possession"
        }
    },
    "Striker, Pressing Forward": {
        "metrics": [
            "Successful defensive actions per 90", "Defensive duels per 90", "Defensive duels won, %",
            "Aerial duels per 90", "Aerial duels won, %", "PAdj Interceptions",
            "Offensive duels per 90", "Offensive duels won, %",
            "Dribbles per 90", "Successful dribbles, %",
            "Forward passes per 90", "Accurate forward passes, %",
            "xA per 90", "Shot assists per 90"
        ],
        "groups": {
            "Successful defensive actions per 90": "Off The Ball",
            "Defensive duels per 90": "Off The Ball",
            "Defensive duels won, %": "Off The Ball",
            "Aerial duels per 90": "Off The Ball",
            "Aerial duels won, %": "Off The Ball",
            "PAdj Interceptions": "Off The Ball",
            "Offensive duels per 90": "Possession",
            "Offensive duels won, %": "Possession",
            "Dribbles per 90": "Possession",
            "Successful dribbles, %": "Possession",
            "Forward passes per 90": "Possession",
            "Accurate forward passes, %": "Possession",
            "xA per 90": "Attacking",
            "Shot assists per 90": "Attacking"
        }
    }
}

# Colors for genre backgrounds / labels
group_colors = {
    "Off The Ball": "#8b5cf6",  # purple
    "Attacking":    "#3b82f6",  # blue
    "Possession":   "#f9a8d4",  # pink
    "Defensive":    "#6366f1",  # indigo
    "Goalkeeping":  "#a16207",  # cyan
}

# ================== File upload ==================
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if not uploaded_file:
    st.stop()

df = pd.read_excel(uploaded_file)

# ================== Position helpers/filters ==================
if "Position" in df.columns:
    df["Positions played"] = df["Position"].astype(str)
else:
    df["Positions played"] = np.nan

df["Six-Group Position"] = df["Position"].apply(map_first_position_to_group) if "Position" in df.columns else np.nan

# Minutes
minutes_col = "Minutes played"
min_minutes = st.number_input("Minimum minutes to include", min_value=0, value=1000, step=50)
df["_minutes_numeric"] = pd.to_numeric(df.get(minutes_col, np.nan), errors="coerce")
df = df[df["_minutes_numeric"] >= min_minutes].copy()
if df.empty:
    st.warning("No players meet the minutes threshold. Lower the minimum.")
    st.stop()

# Age
if "Age" in df.columns:
    df["_age_numeric"] = pd.to_numeric(df["Age"], errors="coerce")
    if df["_age_numeric"].notna().any():
        age_min = int(np.nanmin(df["_age_numeric"]))
        age_max = int(np.nanmax(df["_age_numeric"]))
        sel_min, sel_max = st.slider("Age range to include",
                                     min_value=age_min, max_value=age_max,
                                     value=(age_min, age_max), step=1)
        df = df[df["_age_numeric"].between(sel_min, sel_max)].copy()
    else:
        st.info("Age column has no numeric values, age filter skipped.")
else:
    st.info("No Age column found, age filter skipped.")

st.caption(f"Filtering on '{minutes_col}' ≥ {min_minutes}. Players remaining, {len(df)}")

# 6-group include filter
available_groups = [g for g in SIX_GROUPS if g in df["Six-Group Position"].unique()]
selected_groups = st.multiselect("Include groups", options=available_groups, default=[], label_visibility="collapsed")
if selected_groups:
    df = df[df["Six-Group Position"].isin(selected_groups)].copy()
    if df.empty:
        st.warning("No players after 6-group filter. Clear filters or choose different groups.")
        st.stop()

# Track if exactly one group is selected
current_single_group = selected_groups[0] if len(selected_groups) == 1 else None

# ================== Session state for selections ==================
if "selected_player" not in st.session_state:
    st.session_state.selected_player = None
if "selected_template" not in st.session_state:
    st.session_state.selected_template = None
if "last_auto_group" not in st.session_state:
    st.session_state.last_auto_group = None
if "ec_rows" not in st.session_state:
    st.session_state.ec_rows = 1

# Initialise template once (prefer single-group default)
if st.session_state.selected_template is None:
    if current_single_group:
        st.session_state.selected_template = DEFAULT_TEMPLATE.get(current_single_group, list(position_metrics.keys())[0])
        st.session_state.last_auto_group = current_single_group
    else:
        st.session_state.selected_template = list(position_metrics.keys())[0]

# If the 6-group selection changed to a *new* single group, snap to that default
if current_single_group is not None and current_single_group != st.session_state.last_auto_group:
    st.session_state.selected_template = DEFAULT_TEMPLATE.get(current_single_group, st.session_state.selected_template)
    st.session_state.last_auto_group = current_single_group

# ================== Build metric pool & EC ==================
current_template_name = st.session_state.selected_template or list(position_metrics.keys())[0]
current_metrics = position_metrics[current_template_name]["metrics"]
for m in current_metrics:
    if m not in df.columns:
        df[m] = 0
df[current_metrics] = df[current_metrics].fillna(0)

with st.expander("Essential Criteria", expanded=False):
    use_all_cols = st.checkbox("Pick from all numeric columns", value=False,
                               help="Unchecked, only metrics in the selected template are shown")
    numeric_cols_all = sorted([c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])
    metric_pool_base = numeric_cols_all if use_all_cols else current_metrics

    cbtn1, cbtn2, cbtn3 = st.columns(3)
    with cbtn1:
        if st.button("Add criterion"):
            st.session_state.ec_rows += 1
    with cbtn2:
        if st.button("Remove last", disabled=st.session_state.ec_rows <= 1):
            st.session_state.ec_rows = max(1, st.session_state.ec_rows - 1)
    with cbtn3:
        apply_nonneg = st.checkbox("Apply all criteria", value=False)

    if len(metric_pool_base) == 0:
        st.info("No numeric metrics available to filter.")
        apply_nonneg = False
        st.session_state.ec_rows = 1

    criteria = []
    for i in range(st.session_state.ec_rows):
        st.markdown(f"**Criterion {i+1}**")
        c1, c2, c3, c4 = st.columns([3, 2, 2, 3])

        prev_key_metric = f"ec_metric_{i}"
        prev_metric = st.session_state.get(prev_key_metric, None)
        metric_pool_display = list(metric_pool_base)
        if prev_metric and prev_metric not in metric_pool_display and prev_metric in numeric_cols_all:
            metric_pool_display = [prev_metric] + [m for m in metric_pool_display if m != prev_metric]

        with c1:
            metric_name = st.selectbox("Metric", metric_pool_display, key=prev_key_metric)
        with c2:
            mode = st.radio("Apply to", ["Raw", "Percentile"], horizontal=True, key=f"ec_mode_{i}")
        with c3:
            op = st.selectbox("Operator", [">=", ">", "<=", "<"], index=0, key=f"ec_op_{i}")
        with c4:
            if mode == "Percentile":
                default_thr = 50.0
            else:
                default_thr = float(np.nanmedian(pd.to_numeric(df[metric_name], errors="coerce")))
                if not np.isfinite(default_thr):
                    default_thr = 0.0
            thr_str = st.text_input("Threshold", value=str(int(default_thr)), key=f"ec_thr_{i}")
            try:
                thr_val = float(thr_str)
            except ValueError:
                thr_val = default_thr
        criteria.append((metric_name, mode, op, thr_val))

    if apply_nonneg and len(criteria) > 0:
        temp_cols = []
        mask_all = pd.Series(True, index=df.index)

        for metric_name, mode, op, thr_val in criteria:
            if mode == "Percentile":
                df[metric_name] = pd.to_numeric(df[metric_name], errors="coerce")
                perc_series = (df[metric_name].rank(pct=True) * 100).round(1)
                tmp_col = f"__tmp_percentile__{metric_name}"
                df[tmp_col] = perc_series
                filter_col = tmp_col
                temp_cols.append(tmp_col)
            else:
                filter_col = metric_name
                df[filter_col] = pd.to_numeric(df[filter_col], errors="coerce")

            if op == ">=":
                mask = df[filter_col] >= thr_val
            elif op == ">":
                mask = df[filter_col] > thr_val
            elif op == "<=":
                mask = df[filter_col] <= thr_val
            else:
                mask = df[filter_col] < thr_val

            mask_all &= mask

        kept = int(mask_all.sum())
        dropped = int((~mask_all).sum())
        df = df[mask_all].copy()

        if temp_cols:
            df.drop(columns=temp_cols, inplace=True, errors="ignore")

        summary = " AND ".join(
            [f"{m} {o} {t}{'%' if md=='Percentile' else ''}" for m, md, o, t in criteria]
        )
        st.caption(f"Essential Criteria applied: {summary}. Kept {kept}, removed {dropped} players.")

# ================== Prepare data for plotting ==================
metrics = position_metrics[current_template_name]["metrics"]
metric_groups = position_metrics[current_template_name]["groups"]

for m in metrics:
    if m not in df.columns:
        df[m] = 0
df[metrics] = df[metrics].fillna(0)

metrics_df = df[metrics].copy()
percentile_df = (metrics_df.rank(pct=True) * 100).round(1)

keep_cols = ["Player", "Team within selected timeframe", "Team", "Age", "Height", "Positions played", "Minutes played"]
for c in keep_cols:
    if c not in df.columns:
        df[c] = np.nan

plot_data = pd.concat([df[keep_cols], metrics_df, percentile_df.add_suffix(" (percentile)")], axis=1)

sel_metrics = list(metric_groups.keys())
percentiles_all = plot_data[[m + " (percentile)" for m in sel_metrics]]
z_scores_all = (percentiles_all - 50) / 15
plot_data["Avg Z Score"] = z_scores_all.mean(axis=1)
plot_data["Rank"] = plot_data["Avg Z Score"].rank(ascending=False, method="min").astype(int)

# ================== Player & template selectors ==================
players = df["Player"].dropna().unique().tolist()
if not players:
    st.warning("No players available after filters.")
    st.stop()

if st.session_state.selected_player not in players:
    st.session_state.selected_player = players[0]

selected_player = st.selectbox(
    "Choose a player",
    players,
    index=players.index(st.session_state.selected_player) if st.session_state.selected_player in players else 0,
    key="player_select"
)
st.session_state.selected_player = selected_player

template_names = list(position_metrics.keys())
tpl_index = template_names.index(current_template_name) if current_template_name in template_names else 0
selected_position_template = st.selectbox(
    "Choose a position template for the chart",
    template_names,
    index=tpl_index,
    key="template_select"
)
st.session_state.selected_template = selected_position_template

# If template changed, recompute metric sets quickly
if selected_position_template != current_template_name:
    st.rerun()

# ================== Chart ==================
def plot_radial_bar_grouped(player_name, plot_data, metric_groups, group_colors):
    row = plot_data[plot_data["Player"] == player_name]
    if row.empty:
        st.error(f"No player named '{player_name}' found.")
        return

    # order & data for the chosen template
    sel_metrics_loc = list(metric_groups.keys())
    raw         = row[sel_metrics_loc].values.flatten()
    percentiles = row[[m + " (percentile)" for m in sel_metrics_loc]].values.flatten()
    groups      = [metric_groups[m] for m in sel_metrics_loc]

    n = len(sel_metrics_loc)
    if n == 0:
        st.warning("No metrics available for this template.")
        return

    step   = 2 * np.pi / n
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)

    # bar colours from percentile (your red→orange→green gradients by tercile)
    bar_colors = [percentile_to_color(p) for p in percentiles]

    # figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)

    # keep scale 0..100; hide the outer ring by not placing a tick at 100
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80])      # <- no 100 tick => no outer ring line
    ax.yaxis.grid(True, alpha=0.12)
    ax.set_xticks([])                     # we'll place metric labels manually
    ax.spines["polar"].set_visible(False)
    ax.patch.set_linewidth(0)

    # ===== Background wedges (contiguous runs per genre), coloured by AVG percentile =====
    # Make wedges align to the FIRST/LAST bar of a run, and trim edges to create visible gaps.
    bar_width = step * 0.90               # MUST match bar() width below
    bar_half  = bar_width / 2.0
    gap_each_side = step * SECTION_GAP_FRAC  # angular trim on EACH side of a section wedge

    # Build contiguous runs of the same genre around the circle
    runs, run_start = [], 0
    for i in range(1, n):
        if groups[i] != groups[i - 1]:
            runs.append((run_start, i - 1, groups[i - 1]))
            run_start = i
    runs.append((run_start, n - 1, groups[-1]))

    def _draw_wedge(edge_start, edge_end, color, label_text=None, label_r=None):
        """Fill a wedge between two angular edges; optional label outside."""
        width  = edge_end - edge_start
        center = edge_start + width / 2.0
        ax.bar([center], [100], width=width, bottom=0,
               color=color, alpha=GENRE_BG_ALPHA,
               edgecolor=None, linewidth=0, zorder=0)
        if label_text is not None and label_r is not None:
            ax.text(center, label_r, label_text,
                    ha="center", va="center",
                    rotation=0, rotation_mode="anchor",
                    fontsize=12, fontweight="bold",
                    color="black", zorder=20, clip_on=False)  # plain black text

    two_pi = 2 * np.pi
    for start_idx, end_idx, g in runs:
        # Average percentile across this section -> background colour
        run_avg = float(np.nanmean(percentiles[start_idx:end_idx+1]))
        bg_color = percentile_to_color(run_avg)

        # Wedge edges aligned to bars, then trimmed to make the section gap larger
        start_edge = (angles[start_idx] - bar_half) + gap_each_side
        end_edge   = (angles[end_idx]   + bar_half) - gap_each_side

        # In case a tiny run is narrower than 2*gap, clamp minimally
        if end_edge - start_edge <= step * 0.10:
            mid = (start_edge + end_edge) / 2.0
            start_edge = mid - step * 0.05
            end_edge   = mid + step * 0.05

        # Normalise and handle wrap-around
        while start_edge < 0:
            start_edge += two_pi
            end_edge   += two_pi
        while end_edge <= start_edge:
            end_edge += two_pi

        # Draw (split if crossing 2π) and place the black label on the head piece
        if end_edge > two_pi:
            _draw_wedge(start_edge, two_pi, bg_color)  # tail
            _draw_wedge(0.0, end_edge - two_pi, bg_color, g, GENRE_LABEL_R)  # head + label
        else:
            _draw_wedge(start_edge, end_edge, bg_color, g, GENRE_LABEL_R)

    # ===== Bars (percentiles) =====
    ax.bar(
        angles, percentiles,
        width=bar_width,
        color=bar_colors, edgecolor=bar_colors,
        alpha=0.95, zorder=10
    )

    # raw values printed at mid radius
    for ang, val in zip(angles, raw):
        try:
            txt = f"{float(val):.2f}"
        except Exception:
            txt = "-"
        ax.text(ang, 50, txt,
                ha="center", va="center",
                color="black", fontsize=10, fontweight="bold",
                zorder=15)

    # metric labels just inside the outer ring
    for i, ang in enumerate(angles):
        label = sel_metrics_loc[i].replace(" per 90", "").replace(", %", " (%)")
        ax.text(ang, 104, label,
                ha="center", va="center",
                color="black", fontsize=10, fontweight="bold",
                zorder=15)

    # ===== Title & badge (unchanged) =====
    age      = row["Age"].values[0]
    height   = row["Height"].values[0]
    team     = row["Team within selected timeframe"].values[0]
    mins     = row["Minutes played"].values[0] if "Minutes played" in row else np.nan
    rank_val = int(row["Rank"].values[0]) if "Rank" in row else None

    age_str    = f"{int(age)} years old" if not pd.isnull(age) else ""
    height_str = f"{int(height)} cm"     if not pd.isnull(height) else ""
    parts = [player_name] + [p for p in (age_str, height_str) if p]
    line1 = " | ".join(parts)

    team_str = f"{team}" if pd.notnull(team) else ""
    mins_str = f"{int(mins)} mins" if pd.notnull(mins) else ""
    rank_str = f"Rank #{rank_val}" if rank_val is not None else ""
    line2 = " | ".join([p for p in (team_str, mins_str, rank_str) if p])

    ax.set_title(f"{line1}\n{line2}", color="black", size=22, pad=20, y=1.12)

    z_scores = (percentiles - 50) / 15
    avg_z = float(np.nanmean(z_scores))
    if   avg_z >= 1.0:  badge = ("Excellent", "#228B22")
    elif avg_z >= 0.3:  badge = ("Good",      "#1E90FF")
    elif avg_z >= -0.3: badge = ("Average",   "#DAA520")
    else:               badge = ("Below Average", "#DC143C")

    st.markdown(
        f"<div style='text-align:center; margin-top: 20px;'>"
        f"<span style='font-size:24px; font-weight:bold;'>Average Z Score, {avg_z:.2f}</span><br>"
        f"<span style='background-color:{badge[1]}; color:white; padding:5px 10px; border-radius:8px; font-size:20px;'>{badge[0]}</span></div>",
        unsafe_allow_html=True
    )

    st.pyplot(fig)

# draw chart
if st.session_state.selected_player:
    plot_radial_bar_grouped(st.session_state.selected_player, plot_data, metric_groups, group_colors)

# ================== Ranking table ==================
st.markdown("### Players Ranked by Z-Score")
cols_for_table = ["Player", "Positions played", "Age", "Team", "Team within selected timeframe",
                  "Minutes played", "Avg Z Score", "Rank"]
z_ranking = (plot_data[cols_for_table]
             .sort_values(by="Avg Z Score", ascending=False)
             .reset_index(drop=True))
z_ranking[["Team", "Team within selected timeframe"]] = z_ranking[["Team", "Team within selected timeframe"]].fillna("N/A")
if "Age" in z_ranking:
    z_ranking["Age"] = z_ranking["Age"].apply(lambda x: int(x) if pd.notnull(x) else x)
z_ranking.index = np.arange(1, len(z_ranking) + 1)
z_ranking.index.name = "Row"

st.dataframe(z_ranking, use_container_width=True)
