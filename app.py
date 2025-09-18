import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# --- Basic password protection ---
PASSWORD = "Livi2025"

st.set_page_config(page_title="Livingston FC Recruitment App", layout="centered")

# ---------- Club Branding Row ----------
col1, col2, col3 = st.columns([1, 4, 1])  # adjust ratios to balance width

with col1:
    st.image("assets/Livingston_FC_club_badge_new.png", width=120)  # left logo

with col2:
    st.markdown(
        "<h1 style='text-align: center;'>Livingston FC Recruitment App</h1>",
        unsafe_allow_html=True
    )

with col3:
    st.image("assets/club_logo.png", width=120)  # right logo

# Ask for password
pwd = st.text_input("Enter password:", type="password")
if pwd != PASSWORD:
    st.warning("Please enter the correct password to access the app.")
    st.stop()

# ========== 6-position mapping ==========
SIX_GROUPS = [
    "Goalkeeper",
    "Wide Defender",
    "Central Defender",
    "Central Midfielder",
    "Wide Midfielder",
    "Central Forward"
]

RAW_TO_SIX = {
    # Goalkeeper
    "GK": "Goalkeeper", "GKP": "Goalkeeper", "GOALKEEPER": "Goalkeeper",

    # Wide Defender
    "RB": "Wide Defender", "LB": "Wide Defender",
    "RWB": "Wide Defender", "LWB": "Wide Defender", "RFB": "Wide Defender", "LFB": "Wide Defender",

    # Central Defender
    "CB": "Central Defender", "RCB": "Central Defender", "LCB": "Central Defender",
    "CBR": "Central Defender", "CBL": "Central Defender", "SW": "Central Defender",

    # Central Midfielder
    "CMF": "Central Midfielder", "CM": "Central Midfielder",
    "RCMF": "Central Midfielder", "RCM": "Central Midfielder",
    "LCMF": "Central Midfielder", "LCM": "Central Midfielder",
    "DMF": "Central Midfielder", "DM": "Central Midfielder", "CDM": "Central Midfielder",
    "RDMF": "Central Midfielder", "RDM": "Central Midfielder",
    "LDMF": "Central Midfielder", "LDM": "Central Midfielder",
    "AMF": "Central Midfielder", "AM": "Central Midfielder", "CAM": "Central Midfielder",
    "SS": "Central Midfielder", "10": "Central Midfielder",

    # Wide Midfielder
    "LWF": "Wide Midfielder", "RWF": "Wide Midfielder",
    "RW": "Wide Midfielder", "LW": "Wide Midfielder",
    "LAMF": "Wide Midfielder", "RAMF": "Wide Midfielder",
    "RM": "Wide Midfielder", "LM": "Wide Midfielder",
    "WF": "Wide Midfielder", "RWG": "Wide Midfielder", "LWG": "Wide Midfielder", "W": "Wide Midfielder",

    # Central Forward
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

# ========== Default template mapping (must match position_metrics keys) ==========
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

# Colors
group_colors = {
    "Off The Ball": "crimson",
    "Attacking": "royalblue",
    "Possession": "seagreen",
    "Defensive": "darkorange",
    "Goalkeeping": "purple"
}

# ---------- File upload ----------
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if not uploaded_file:
    st.stop()

df = pd.read_excel(uploaded_file)

# ---------- Positions ----------
if "Position" in df.columns:
    df["Positions played"] = df["Position"].astype(str)
else:
    df["Positions played"] = np.nan

df["Six-Group Position"] = df["Position"].apply(map_first_position_to_group) if "Position" in df.columns else np.nan

# ---------- Minutes filter ----------
minutes_col = "Minutes played"
min_minutes = st.number_input("Minimum minutes to include", min_value=0, value=1000, step=50)
df["_minutes_numeric"] = pd.to_numeric(df.get(minutes_col, np.nan), errors="coerce")
df = df[df["_minutes_numeric"] >= min_minutes].copy()
if df.empty:
    st.warning("No players meet the minutes threshold. Lower the minimum.")
    st.stop()

# ---------- Age slider filter ----------
if "Age" in df.columns:
    df["_age_numeric"] = pd.to_numeric(df["Age"], errors="coerce")
    if df["_age_numeric"].notna().any():
        age_min = int(np.nanmin(df["_age_numeric"]))
        age_max = int(np.nanmax(df["_age_numeric"]))
        sel_min, sel_max = st.slider("Age range to include", min_value=age_min, max_value=age_max, value=(age_min, age_max), step=1)
        df = df[df["_age_numeric"].between(sel_min, sel_max)].copy()
    else:
        st.info("Age column has no numeric values, age filter skipped.")
else:
    st.info("No Age column found, age filter skipped.")

st.caption(f"Filtering on '{minutes_col}' ≥ {min_minutes}. Players remaining, {len(df)}")

# ---------- 6-group filter ----------
available_groups = [g for g in SIX_GROUPS if g in df["Six-Group Position"].unique()]
selected_groups = st.multiselect("Include groups", options=available_groups, default=[], label_visibility="collapsed")
if selected_groups:
    df = df[df["Six-Group Position"].isin(selected_groups)].copy()
    if df.empty:
        st.warning("No players after 6-group filter. Clear filters or choose different groups.")
        st.stop()

# Track if exactly one group is selected
current_single_group = selected_groups[0] if len(selected_groups) == 1 else None

# ---------- Session state for player/template & EC ----------
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
# If zero/multiple groups, don't auto-change the template.

# ---------- Build metric pool for Essential Criteria (needs current template) ----------
current_template_name = st.session_state.selected_template or list(position_metrics.keys())[0]
current_metrics = position_metrics[current_template_name]["metrics"]
for m in current_metrics:
    if m not in df.columns:
        df[m] = 0
df[current_metrics] = df[current_metrics].fillna(0)

# ---------- Essential Criteria (multiple AND rules) ----------
with st.expander("Essential Criteria", expanded=False):
    use_all_cols = st.checkbox(
        "Pick from all numeric columns",
        value=False,
        help="Unchecked, only metrics in the selected template are shown"
    )
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
        # Keep previous metric visible even if pool changed (prevents jumpiness)
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

# ---------- Player select (after EC). Changing player NEVER changes template ----------
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

# ---------- Template select (user-controlled). Only auto-snaps when single-group changes ----------
template_names = list(position_metrics.keys())
tpl_index = template_names.index(st.session_state.selected_template) if st.session_state.selected_template in template_names else 0
selected_position_template = st.selectbox(
    "Choose a position template for the chart",
    template_names,
    index=tpl_index,
    key="template_select"
)
st.session_state.selected_template = selected_position_template

# ---------- Metrics for selected template, then percentiles ----------
metrics = position_metrics[selected_position_template]["metrics"]
metric_groups = position_metrics[selected_position_template]["groups"]

# Ensure columns exist + fill
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

# ---------- Chart ----------
def plot_radial_bar_grouped(player_name, plot_data, metric_groups, group_colors):
    row = plot_data[plot_data["Player"] == player_name]
    if row.empty:
        st.error(f"No player named '{player_name}' found.")
        return

    sel_metrics_loc = list(metric_groups.keys())
    raw = row[sel_metrics_loc].values.flatten()
    percentiles = row[[m + " (percentile)" for m in sel_metrics_loc]].values.flatten()
    groups = [metric_groups[m] for m in sel_metrics_loc]
    colors = [group_colors.get(g, "grey") for g in groups]

    num_bars = len(sel_metrics_loc)
    angles = np.linspace(0, 2*np.pi, num_bars, endpoint=False)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)

    ax.bar(angles, percentiles, width=2*np.pi/num_bars*0.9, color=colors, edgecolor=colors, alpha=0.75)

    for angle, raw_val in zip(angles, raw):
        ax.text(angle, 50, f"{raw_val:.2f}", ha="center", va="center", color="black", fontsize=10, fontweight="bold")

    for i, angle in enumerate(angles):
        label = sel_metrics_loc[i].replace(" per 90", "").replace(", %", " (%)")
        ax.text(angle, 108, label, ha="center", va="center", color="black", fontsize=10, fontweight="bold")

    group_positions = {}
    for g, a in zip(groups, angles):
        group_positions.setdefault(g, []).append(a)
    for group, group_angles in group_positions.items():
        mean_angle = np.mean(group_angles)
        ax.text(mean_angle, 125, group, ha="center", va="center", fontsize=20, fontweight="bold", color=group_colors.get(group, "grey"))

    age = row["Age"].values[0]
    height = row["Height"].values[0]
    team = row["Team within selected timeframe"].values[0]
    mins = row["Minutes played"].values[0] if "Minutes played" in row else np.nan
    rank_val = int(row["Rank"].values[0]) if "Rank" in row else None

    age_str = f"{int(age)} years old" if not pd.isnull(age) else ""
    height_str = f"{int(height)} cm" if not pd.isnull(height) else ""
    parts = [player_name]
    if age_str: parts.append(age_str)
    if height_str: parts.append(height_str)
    line1 = " | ".join(parts)

    team_str = f"{team}" if pd.notnull(team) else ""
    mins_str = f"{int(mins)} mins" if pd.notnull(mins) else ""
    rank_str = f"Rank #{rank_val}" if rank_val is not None else ""
    line2 = " | ".join([p for p in [team_str, mins_str, rank_str] if p])

    ax.set_title(f"{line1}\n{line2}", color="black", size=22, pad=20, y=1.12)

    z_scores = (percentiles - 50) / 15
    avg_z = np.mean(z_scores)

    if avg_z >= 1.0:
        badge = ("Excellent", "#228B22")
    elif avg_z >= 0.3:
        badge = ("Good", "#1E90FF")
    elif avg_z >= -0.3:
        badge = ("Average", "#DAA520")
    else:
        badge = ("Below Average", "#DC143C")

    st.markdown(
        f"<div style='text-align:center; margin-top: 20px;'>"
        f"<span style='font-size:24px; font-weight:bold;'>Average Z Score, {avg_z:.2f}</span><br>"
        f"<span style='background-color:{badge[1]}; color:white; padding:5px 10px; border-radius:8px; font-size:20px;'>{badge[0]}</span></div>",
        unsafe_allow_html=True
    )

    st.pyplot(fig, use_container_width=True)

if st.session_state.selected_player:
    plot_radial_bar_grouped(st.session_state.selected_player, plot_data, metric_groups, group_colors)

# ---------- Ranking table ----------
st.markdown("### Players Ranked by Z-Score")
cols_for_table = ["Player", "Positions played", "Age", "Team", "Team within selected timeframe", "Minutes played", "Avg Z Score", "Rank"]
z_ranking = (plot_data[cols_for_table].sort_values(by="Avg Z Score", ascending=False).reset_index(drop=True))
z_ranking[["Team", "Team within selected timeframe"]] = z_ranking[["Team", "Team within selected timeframe"]].fillna("N/A")
if "Age" in z_ranking:
    z_ranking["Age"] = z_ranking["Age"].apply(lambda x: int(x) if pd.notnull(x) else x)
z_ranking.index = np.arange(1, len(z_ranking) + 1)
z_ranking.index.name = "Row"

st.dataframe(z_ranking, use_container_width=True)
