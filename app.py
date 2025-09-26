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

# ---------- Fixed group colours ----------
group_colors = {
    "Attacking":   "crimson",
    "Possession":  "seagreen",
    "Defensive":   "royalblue",
    "Goalkeeping": "purple",
}

# --- League name normalisation: StatsBomb -> your Opta names ---
LEAGUE_SYNONYMS = {
    "A-League": "Australia A-League Men",
    "2. Liga": "Austria 2. Liga",
    "Challenger Pro League": "Belgium Challenger Pro League",
    "First League": "Bulgaria First League",
    "1. HNL": "Croatia 1. HNL", "HNL": "Croatia 1. HNL",
    "Czech Liga": "Czech First Tier",
    "1st Division": "Denmark 1st Division", "Superliga": "Denmark Superliga",
    "League One": "England League One", "League Two": "England League Two",
    "National League": "England National League", "National League N / S": "England National League N/S",
    "Premium Liiga": "Estonia Premium Liiga",
    "Veikkausliiga": "Finland Veikkausliiga",
    "Championnat National": "France National 1",
    "3. Liga": "Germany 3. Liga",
    "Super League": "Greece Super League 1",
    "NB I": "Hungary NB I",
    "Besta deild karla": "Iceland Besta Deild",
    "Serie C": "Italy Serie C",
    "J2 League": "Japan J2 League",
    "Virsliga": "Latvia Virsliga",
    "A Lyga": "Lithuania A Lyga",
    "Botola Pro": "Morocco Botola Pro",
    "Eerste Divisie": "Netherlands Eerste Divisie",
    "1. Division": "Norway 1. Division", "Eliteserien": "Norway Eliteserien",
    "I Liga": "Poland 1 Liga", "Ekstraklasa": "Poland Ekstraklasa",
    "Segunda Liga": "Portugal Segunda Liga", "Liga Pro": "Portugal Segunda Liga",
    "Premier Division": "Republic of Ireland Premier Division",
    "Liga 1": "Romania Liga 1",
    "Championship": "Scotland Championship", "Premiership": "Scotland Premiership",
    "Super Liga": "Serbia Super Liga",
    "1. Liga": "Slovakia 1. Liga",
    "1. Liga (SVN)": "Slovenia 1. Liga",
    "PSL": "South Africa Premier Division",
    "Allsvenskan": "Sweden Allsvenskan", "Superettan": "Sweden Superettan",
    "Challenge League": "Switzerland Challenge League",
    "Ligue 1": "Tunisia Ligue 1",
    "USL Championship": "USA USL Championship",
}

# ========== Role groups shown in filters ==========
SIX_GROUPS = [
    "Goalkeeper", "Full Back", "Centre Back", "Number 6", "Number 8", "Winger", "Striker"
]

# ========== Position → group mapping ==========
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
    "RIGHTBACK": "Full Back", "LEFTBACK": "Full Back",
    "RIGHTWINGBACK": "Full Back", "LEFTWINGBACK": "Full Back",
    # Centre backs
    "RIGHTCENTREBACK": "Centre Back", "LEFTCENTREBACK": "Centre Back", "CENTREBACK": "Centre Back",
    # Centre mid (generic) → duplicated into 6 & 8 later
    "CENTREMIDFIELDER": "Centre Midfield",
    "RIGHTCENTREMIDFIELDER": "Centre Midfield", "LEFTCENTREMIDFIELDER": "Centre Midfield",
    # Defensive mids → 6
    "DEFENSIVEMIDFIELDER": "Number 6", "RIGHTDEFENSIVEMIDFIELDER": "Number 6", "LEFTDEFENSIVEMIDFIELDER": "Number 6",
    # Attacking mids / 10 → 8
    "CENTREATTACKINGMIDFIELDER": "Number 8", "ATTACKINGMIDFIELDER": "Number 8",
    "SECONDSTRIKER": "Number 8", "10": "Number 8",
    # Wingers / wide mids
    "RIGHTWING": "Winger", "LEFTWING": "Winger",
    "RIGHTMIDFIELDER": "Winger", "LEFTMIDFIELDER": "Winger",
    # Strikers
    "CENTREFORWARD": "Striker", "RIGHTCENTREFORWARD": "Striker", "LEFTCENTREFORWARD": "Striker",
}

def parse_first_position(cell) -> str:
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

# ========== Radar metric sets ==========
position_metrics = {
    # ---------- Goalkeeper ----------
    "Goalkeeper": {
        "metrics": [
            # Possession
            "Pass into Danger%", "Pass into Pressure%",
            # Goalkeeping
            "Goals Conceded", "PSxG Faced", "GSAA", "Save%", "xSv%", "Shot Stopping%",
            "Shots Faced", "Shots Faced OT%", "Positive Outcome%", "Goalkeeper OBV",
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

    # ---------- Centre Back Old ----------
    "Centre Back Old": {
        "metrics": [
            # Attacking
            "xG",
            # Possession
            "Passing%", "Pressured Long Balls", "Unpressured Long Balls", "OBV",
            # Defensive
            "PAdj Interceptions", "PAdj Tackles", "Tack/DP%",
            "Defensive Actions", "Aggressive Actions", "Fouls",
            "Aerial Wins", "Aerial Win%",
        ],
        "groups": {
            "PAdj Interceptions": "Defensive",
            "PAdj Tackles": "Defensive",
            "Tack/DP%": "Defensive",
            "Defensive Actions": "Defensive",
            "Aggressive Actions": "Defensive",
            "Fouls": "Defensive",
            "Aerial Wins": "Defensive",
            "Aerial Win%": "Defensive",
            "Passing%": "Possession",
            "Pressured Long Balls": "Possession",
            "Unpressured Long Balls": "Possession",
            "OBV": "Possession",
            "xG": "Attacking",
        }
    },

    # ---------- Centre Back New ----------
    "Centre Back New": {
        "metrics": [
            # Attacking
            "xG",
            # Possession
            "Passing%", "Pressured Long Balls", "Unpressured Long Balls", "OBV",
            # Defensive
            "PAdj Interceptions", "PAdj Tackles", "Tack/DP%",
            "Defensive Actions", "Aggressive Actions", "Fouls",
            "Aerial Wins", "Aerial Win%",
        ],
        "groups": {
            "PAdj Interceptions": "Defensive",
            "PAdj Tackles": "Defensive",
            "Tack/DP%": "Defensive",
            "Defensive Actions": "Defensive",
            "Aggressive Actions": "Defensive",
            "Fouls": "Defensive",
            "Aerial Wins": "Defensive",
            "Aerial Win%": "Defensive",
            "Passing%": "Possession",
            "Pressured Long Balls": "Possession",
            "Unpressured Long Balls": "Possession",
            "OBV": "Possession",
            "xG": "Attacking",
        }
    },

    # ---------- Full Back Old ----------
    "Full Back Old": {
        "metrics": [
            # Attacking
            "xGBuildup",
            # Possession
            "Passing%", "OP Passes Into Box", "Deep Progressions",
            "Successful Dribbles", "Turnovers", "OBV", "Pass OBV",
            # Defensive
            "Defensive Actions", "Aerial Win%", "PAdj Pressures",
            "PAdj Tack&Int", "Tack/DP%",
        ],
        "groups": {
            "Passing%": "Possession",
            "OP Passes Into Box": "Possession",
            "Deep Progressions": "Possession",
            "xGBuildup": "Attacking",
            "Successful Dribbles": "Possession",
            "Turnovers": "Possession",
            "Defensive Actions": "Defensive",
            "Aerial Win%": "Defensive",
            "PAdj Pressures": "Defensive",
            "PAdj Tack&Int": "Defensive",
            "Tack/DP%": "Defensive",
            "OBV": "Possession",
            "Pass OBV": "Possession",
        }
    },

    # ---------- Full Back New ----------
    "Full Back New": {
        "metrics": [
            # Attacking
            "xGBuildup",
            # Possession
            "Passing%", "OP Passes Into Box", "Deep Progressions",
            "Successful Dribbles", "Turnovers", "OBV", "Pass OBV",
            # Defensive
            "Defensive Actions", "Aerial Win%", "PAdj Pressures",
            "PAdj Tack&Int", "Tack/DP%",
        ],
        "groups": {
            "Passing%": "Possession",
            "OP Passes Into Box": "Possession",
            "Deep Progressions": "Possession",
            "xGBuildup": "Attacking",
            "Successful Dribbles": "Possession",
            "Turnovers": "Possession",
            "Defensive Actions": "Defensive",
            "Aerial Win%": "Defensive",
            "PAdj Pressures": "Defensive",
            "PAdj Tack&Int": "Defensive",
            "Tack/DP%": "Defensive",
            "OBV": "Possession",
            "Pass OBV": "Possession",
        }
    },

    # ---------- Number 6 Old ----------
    "Number 6 Old": {
        "metrics": [
            # Attacking
            "xGBuildup", "xG Assisted",
            # Possession
            "Passing%", "Deep Progressions", "Turnovers", "OBV", "Pass OBV",
            # Defensive
            "PAdj Interceptions", "PAdj Tackles", "Tack/DP%",
            "Aggressive Actions", "Aerial Win%",
        ],
        "groups": {
            "Passing%": "Possession",
            "Deep Progressions": "Possession",
            "xGBuildup": "Attacking",
            "PAdj Interceptions": "Defensive",
            "PAdj Tackles": "Defensive",
            "Tack/DP%": "Defensive",
            "Aggressive Actions": "Defensive",
            "Aerial Win%": "Defensive",
            "Turnovers": "Possession",
            "OBV": "Possession",
            "Pass OBV": "Possession",
            "xG Assisted": "Attacking",
        }
    },

    # ---------- Number 6 New ----------
    "Number 6 New": {
        "metrics": [
            # Attacking
            "xGBuildup", "xG Assisted",
            # Possession
            "Passing%", "Deep Progressions", "Turnovers", "OBV", "Pass OBV",
            # Defensive
            "PAdj Interceptions", "PAdj Tackles", "Tack/DP%",
            "Aggressive Actions", "Aerial Win%",
        ],
        "groups": {
            "Passing%": "Possession",
            "Deep Progressions": "Possession",
            "xGBuildup": "Attacking",
            "PAdj Interceptions": "Defensive",
            "PAdj Tackles": "Defensive",
            "Tack/DP%": "Defensive",
            "Aggressive Actions": "Defensive",
            "Aerial Win%": "Defensive",
            "Turnovers": "Possession",
            "OBV": "Possession",
            "Pass OBV": "Possession",
            "xG Assisted": "Attacking",
        }
    },

    # ---------- Number 8 Old ----------
    "Number 8 Old": {
        "metrics": [
            # Attacking
            "xGBuildup", "xG Assisted", "Shots", "xG",
            # Possession
            "Passing%", "Deep Progressions", "OP Passes Into Box", "Pass OBV", "OBV",
            # Defensive
            "Pressure Regains", "PAdj Pressures", "Opposition Half Ball Recoveries",
            "Aggressive Actions",
        ],
        "groups": {
            "Passing%": "Possession",
            "Deep Progressions": "Possession",
            "xGBuildup": "Attacking",
            "xG Assisted": "Attacking",
            "OP Passes Into Box": "Possession",
            "Pass OBV": "Possession",
            "Shots": "Attacking",
            "xG": "Attacking",
            "Pressure Regains": "Defensive",
            "PAdj Pressures": "Defensive",
            "Opposition Half Ball Recoveries": "Defensive",
            "Aggressive Actions": "Defensive",
            "OBV": "Possession",
        }
    },

    # ---------- Number 8 New ----------
    "Number 8 New": {
        "metrics": [
            # Attacking
            "xGBuildup", "xG Assisted", "Shots", "xG",
            # Possession
            "Passing%", "Deep Progressions", "OP Passes Into Box", "Pass OBV", "OBV",
            # Defensive
            "Pressure Regains", "PAdj Pressures", "Opposition Half Ball Recoveries",
            "Aggressive Actions",
        ],
        "groups": {
            "Passing%": "Possession",
            "Deep Progressions": "Possession",
            "xGBuildup": "Attacking",
            "xG Assisted": "Attacking",
            "OP Passes Into Box": "Possession",
            "Pass OBV": "Possession",
            "Shots": "Attacking",
            "xG": "Attacking",
            "Pressure Regains": "Defensive",
            "PAdj Pressures": "Defensive",
            "Opposition Half Ball Recoveries": "Defensive",
            "Aggressive Actions": "Defensive",
            "OBV": "Possession",
        }
    },

    # ---------- Winger Old ----------
    "Winger Old": {
        "metrics": [
            # Attacking
            "xG", "xG/Shot", "Touches In Box", "OP xG Assisted",
            # Possession
            "OP Passes Into Box", "Successful Box Cross%", "Passing%",
            "Successful Dribbles", "Turnovers", "OBV", "D&C OBV",
            # Defensive
            "Pressure Regains",
        ],
        "groups": {
            "xG": "Attacking",
            "xG/Shot": "Attacking",
            "Touches In Box": "Attacking",
            "OP xG Assisted": "Attacking",
            "OP Passes Into Box": "Possession",
            "Successful Box Cross%": "Possession",
            "Passing%": "Possession",
            "Successful Dribbles": "Possession",
            "Turnovers": "Possession",
            "Pressure Regains": "Defensive",
            "OBV": "Possession",
            "D&C OBV": "Possession",
        }
    },

    # ---------- Winger New ----------
    "Winger New": {
        "metrics": [
            # Attacking
            "xG", "xG/Shot", "Touches In Box", "OP xG Assisted",
            # Possession
            "OP Passes Into Box", "Successful Box Cross%", "Passing%",
            "Successful Dribbles", "Turnovers", "OBV", "D&C OBV",
            # Defensive
            "Pressure Regains",
        ],
        "groups": {
            "xG": "Attacking",
            "xG/Shot": "Attacking",
            "Touches In Box": "Attacking",
            "OP xG Assisted": "Attacking",
            "OP Passes Into Box": "Possession",
            "Successful Box Cross%": "Possession",
            "Passing%": "Possession",
            "Successful Dribbles": "Possession",
            "Turnovers": "Possession",
            "Pressure Regains": "Defensive",
            "OBV": "Possession",
            "D&C OBV": "Possession",
        }
    },

    # ---------- Striker Old ----------
    "Striker Old": {
        "metrics": [
            # Attacking
            "All Goals", "Penalty Goals", "xG", "Shots", "xG/Shot",
            "Shot Touch%", "Touches In Box", "xG Assisted",
            # Possession
            "Fouls Won",
            # Defensive
            "Aerial Win%", "Aerial Wins", "Pressure Regains",
        ],
        "groups": {
            "All Goals": "Attacking",
            "Penalty Goals": "Attacking",
            "xG": "Attacking",
            "Shots": "Attacking",
            "xG/Shot": "Attacking",
            "Shot Touch%": "Attacking",
            "Touches In Box": "Attacking",
            "xG Assisted": "Attacking",
            "Fouls Won": "Possession",
            "Aerial Win%": "Defensive",
            "Aerial Wins": "Defensive",
            "Pressure Regains": "Defensive",
        }
    },

    # ---------- Striker New ----------
    "Striker New": {
        "metrics": [
            # Attacking
            "All Goals", "Penalty Goals", "xG", "Shots", "xG/Shot",
            "Shot Touch%", "Touches In Box", "xG Assisted",
            # Possession
            "Fouls Won",
            # Defensive
            "Aerial Win%", "Aerial Wins", "Pressure Regains",
        ],
        "groups": {
            "All Goals": "Attacking",
            "Penalty Goals": "Attacking",
            "xG": "Attacking",
            "Shots": "Attacking",
            "xG/Shot": "Attacking",
            "Shot Touch%": "Attacking",
            "Touches In Box": "Attacking",
            "xG Assisted": "Attacking",
            "Fouls Won": "Possession",
            "Aerial Win%": "Defensive",
            "Aerial Wins": "Defensive",
            "Pressure Regains": "Defensive",
        }
    },
}

# ---------- Data source: local repo ----------
DATA_PATH = (APP_DIR / "statsbombdata.xlsx")  # or APP_DIR / "statsbombdata" or a folder

def load_one_file(p: Path) -> pd.DataFrame:
    print(f"[DEBUG] Trying to load file at: {p.resolve()}")

    def try_excel() -> pd.DataFrame | None:
        try:
            import openpyxl
            return pd.read_excel(p, engine="openpyxl")
        except ImportError:
            print("[DEBUG] openpyxl not available, trying CSV reader next.")
            return None
        except ValueError as e:
            print(f"[DEBUG] Excel parse failed: {e}. Trying CSV.")
            return None
        except Exception as e:
            print(f"[DEBUG] Excel read raised {type(e).__name__}, trying CSV. {e}")
            return None

    def try_csv() -> pd.DataFrame | None:
        for kwargs in [
            dict(sep=None, engine="python"),
            dict(),
            dict(encoding="latin1"),
        ]:
            try:
                return pd.read_csv(p, **kwargs)
            except Exception:
                continue
        return None

    df = None
    if p.suffix.lower() in {".xlsx", ".xls"}:
        df = try_excel()
        if df is None:
            df = try_csv()
    else:
        df = try_csv()
        if df is None:
            df = try_excel()

    if df is None:
        try:
            with open(p, "rb") as fh:
                head = fh.read(256)
            print(f"[ERROR] Could not read {p.name}. File preview: {head[:256]}")
        except Exception:
            pass
        raise ValueError(f"Unsupported or unreadable file, {p.name}")

    print(f"[DEBUG] Loaded {p.name}, {len(df)} rows, {len(df.columns)} cols")
    return df

def _data_signature(path: Path):
    path = Path(path)
    if path.is_file():
        s = path.stat()
        return ("file", str(path.resolve()), s.st_size, int(s.st_mtime))
    else:
        sigs = []
        for p in sorted(path.iterdir()):
            if p.is_file() and (p.suffix.lower() in {".csv", ".xlsx", ".xls"} or p.suffix == ""):
                try:
                    s = p.stat()
                    sigs.append((str(p.resolve()), s.st_size, int(s.st_mtime)))
                except FileNotFoundError:
                    continue
        return ("dir", str(path.resolve()), tuple(sigs))

@st.cache_data(show_spinner=False)
def load_statsbomb(path: Path, _sig=None) -> pd.DataFrame:
    print(f"[DEBUG] Data path configured as: {path}")
    if not path.exists():
        raise FileNotFoundError(f"statsbombdata not found at {path}. Put a CSV or XLSX there, or a folder of them.")

    if path.is_file():
        return load_one_file(path)

    files = sorted(
        f for f in path.iterdir()
        if f.is_file() and (f.suffix.lower() in {".csv", ".xlsx", ".xls"} or f.suffix == "")
    )
    if not files:
        raise FileNotFoundError(f"No data files found inside {path.name}. Add CSV or XLSX.")

    frames = []
    for f in files:
        try:
            frames.append(load_one_file(f))
        except Exception as e:
            print(f"[WARNING] Skipping {f.name} ({e})")

    if not frames:
        raise ValueError("No readable files found in statsbombdata")

    df = pd.concat(frames, ignore_index=True, sort=False)
    print(f"[DEBUG] Merged {len(files)} files from {path.name}, total rows {len(df)}")
    return df

# ---------- Preprocess DataFrame (define BEFORE it’s used) ----------
def preprocess_df(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    # Normalise Competition name
    if "Competition" in df.columns:
        df["Competition_norm"] = (
            df["Competition"].astype(str).str.strip().map(lambda x: LEAGUE_SYNONYMS.get(x, x))
        )
    else:
        df["Competition_norm"] = np.nan

    # Merge league multipliers (fallback 1.0)
    try:
        multipliers_df = pd.read_excel("league_multipliers.xlsx")
        if {"League", "Multiplier"}.issubset(multipliers_df.columns):
            df = df.merge(multipliers_df, left_on="Competition_norm", right_on="League", how="left")
        else:
            st.warning("league_multipliers.xlsx must have columns: 'League', 'Multiplier'. Using 1.0 for all.")
            df["Multiplier"] = 1.0
    except Exception:
        df["Multiplier"] = 1.0

    # Rename new-provider identifiers
    rename_map = {}
    if "Name" in df.columns: rename_map["Name"] = "Player"
    if "Primary Position" in df.columns: rename_map["Primary Position"] = "Position"
    if "Minutes" in df.columns: rename_map["Minutes"] = "Minutes played"
    df.rename(columns=rename_map, inplace=True)

    # Build "Positions played"
    if "Position" in df.columns:
        if "Secondary Position" in df.columns:
            df["Positions played"] = df["Position"].fillna("").astype(str) + np.where(
                df["Secondary Position"].notna() & (df["Secondary Position"].astype(str) != ""),
                ", " + df["Secondary Position"].astype(str),
                ""
            )
        else:
            df["Positions played"] = df["Position"].astype(str)
    else:
        df["Positions played"] = np.nan

    # Fallbacks
    if "Team within selected timeframe" not in df.columns:
        df["Team within selected timeframe"] = df["Team"] if "Team" in df.columns else np.nan
    if "Height" not in df.columns:
        df["Height"] = np.nan

    # Six-Group mapping (from PRIMARY position only)
    if "Position" in df.columns:
        df["Six-Group Position"] = df["Position"].apply(map_first_position_to_group)
    else:
        df["Six-Group Position"] = np.nan

    # Duplicate generic CMs into both 6 & 8 (baseline only)
    if "Six-Group Position" in df.columns:
        cm_mask = df["Six-Group Position"] == "Centre Midfield"
        if cm_mask.any():
            cm_rows = df.loc[cm_mask].copy()
            cm_as_6 = cm_rows.copy(); cm_as_6["Six-Group Position"] = "Number 6"
            cm_as_8 = cm_rows.copy(); cm_as_8["Six-Group Position"] = "Number 8"
            df = pd.concat([df, cm_as_6, cm_as_8], ignore_index=True)

    return df

# ---------- Load & preprocess ----------
df_all_raw = load_statsbomb(DATA_PATH, _sig=_data_signature(DATA_PATH))
# ---------- Clean raw column headers (do this immediately after loading) ----------
# Removes stray spaces and non-breaking spaces so column lookups match
df_all_raw.columns = (
    df_all_raw.columns.astype(str)
    .str.strip()                              # trim leading/trailing spaces
    .str.replace(u"\xa0", " ", regex=False)   # replace non-breaking spaces with normal spaces
    .str.replace(r"\s+", " ", regex=True)     # collapse multiple spaces to a single space
)

# (Optional) quick debug to verify key columns are present exactly as expected
print("[DEBUG] First 10 cleaned columns:", list(df_all_raw.columns[:10]))
print("[DEBUG] Has 'Successful Box Cross%':", "Successful Box Cross%" in df_all_raw.columns)
df_all = preprocess_df(df_all_raw)   # baseline (full dataset, fully prepared)
df = df_all.copy()                   # working copy (filtered by UI)

# ---------- League filter ----------
league_col = "Competition_norm" if "Competition_norm" in df.columns else "Competition"
if league_col not in df.columns:
    df[league_col] = np.nan

df[league_col] = df[league_col].astype(str).str.strip()
all_leagues = sorted([x for x in df[league_col].dropna().unique() if x != ""])

st.markdown("### Choose league (multiple allowed)")

if "league_selection" not in st.session_state:
    st.session_state.league_selection = all_leagues.copy()

b1, b2, _ = st.columns([1, 1, 6])
with b1:
    if st.button("Select all"):
        st.session_state.league_selection = all_leagues.copy()
with b2:
    if st.button("Clear all"):
        st.session_state.league_selection = []

selected_leagues = st.multiselect(
    "Leagues to include",
    options=all_leagues,
    default=st.session_state.league_selection,
    key="league_selection",
)

if selected_leagues:
    df = df[df[league_col].isin(selected_leagues)].copy()
    st.caption(f"Leagues selected: {len(selected_leagues)} | Players: {len(df)}")
    if df.empty:
        st.warning("No players match the selected leagues. Clear or change the league filter.")
        st.stop()
else:
    st.info("No leagues selected. Pick at least one or click ‘Select all’.")
    st.stop()

# ---------- Minutes filter ----------
minutes_col = "Minutes played"
if minutes_col not in df.columns:
    df[minutes_col] = np.nan

min_minutes = st.number_input("Minimum minutes to include", min_value=0, value=1000, step=50)
df["_minutes_numeric"] = pd.to_numeric(df[minutes_col], errors="coerce")
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

# ---------- Group filter ----------
available_groups = [g for g in SIX_GROUPS if "Six-Group Position" in df.columns and g in df["Six-Group Position"].unique()]
selected_groups = st.multiselect("Include groups", options=available_groups, default=[], label_visibility="collapsed")
if selected_groups:
    df = df[df["Six-Group Position"].isin(selected_groups)].copy()
    if df.empty:
        st.warning("No players after group filter. Clear filters or choose different groups.")
        st.stop()

current_single_group = selected_groups[0] if len(selected_groups) == 1 else None

# ---------- Session state ----------
if "selected_player" not in st.session_state:
    st.session_state.selected_player = None
if "ec_rows" not in st.session_state:
    st.session_state.ec_rows = 1
if "template_select" not in st.session_state:
    st.session_state.template_select = list(position_metrics.keys())[0]
if "last_template_choice" not in st.session_state:
    st.session_state.last_template_choice = st.session_state.template_select
if "manual_override" not in st.session_state:
    st.session_state.manual_override = False
if "auto_just_applied" not in st.session_state:
    st.session_state.auto_just_applied = False
if "last_player_for_auto" not in st.session_state:
    st.session_state.last_player_for_auto = None
if "last_groups_tuple" not in st.session_state:
    st.session_state.last_groups_tuple = tuple(selected_groups)

if tuple(selected_groups) != st.session_state.last_groups_tuple:
    if len(selected_groups) == 1:
        st.session_state.manual_override = False
    st.session_state.last_groups_tuple = tuple(selected_groups)

# ---------- Build metric pool for Essential Criteria ----------
current_template_name = st.session_state.template_select or list(position_metrics.keys())[0]
current_metrics = position_metrics[current_template_name]["metrics"]

for m in current_metrics:
    if m not in df.columns:
        df[m] = 0
df[current_metrics] = df[current_metrics].fillna(0)

# ---------- Essential Criteria ----------
with st.expander("Essential Criteria", expanded=False):
    use_all_cols = st.checkbox("Pick from all numeric columns", value=False)
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

        summary = " AND ".join([f"{m} {o} {t}{'%' if md=='Percentile' else ''}" for m, md, o, t in criteria])
        st.caption(f"Essential Criteria applied: {summary}. Kept {kept}, removed {dropped} players.")

# ---------- Player list ----------
if "Player" not in df.columns:
    st.error("Expected a 'Name' column in the upload (renamed to 'Player').")
    st.stop()

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

# ---------- Template select ----------
template_names = list(position_metrics.keys())
idx = template_names.index(st.session_state.template_select) if st.session_state.template_select in template_names else 0
selected_position_template = st.selectbox(
    "Choose a position template for the chart",
    template_names,
    index=idx,
    key="template_select",
)

if st.session_state.auto_just_applied:
    st.session_state.last_template_choice = st.session_state.template_select
    st.session_state.auto_just_applied = False
else:
    if st.session_state.template_select != st.session_state.last_template_choice:
        st.session_state.manual_override = True
        st.session_state.last_template_choice = st.session_state.template_select

# ---------- Metrics + percentiles ----------
metrics = position_metrics[selected_position_template]["metrics"]
metric_groups = position_metrics[selected_position_template]["groups"]

# ensure columns exist in both df_all and df
for m in metrics:
    if m not in df_all.columns: df_all[m] = 0
    if m not in df.columns:     df[m] = 0
df_all[metrics] = df_all[metrics].fillna(0)
df[metrics]     = df[metrics].fillna(0)

# Metrics where lower values are better (raw values unchanged; only percentiles invert)
LOWER_IS_BETTER = {
    "Turnovers",
    "Fouls",
    "Pressured Long Balls",
    "Unpressured Long Balls",
}

def pct_rank(series: pd.Series, lower_is_better: bool) -> pd.Series:
    # pandas: rank(pct=True, ascending=True) -> smallest ≈ 0, largest = 1.0
    r = series.rank(pct=True, ascending=True)

    if lower_is_better:
        p = 1.0 - r          # smaller raw -> larger percentile
    else:
        p = r                # larger raw -> larger percentile

    return (p * 100.0).round(1)

# --- A) Percentiles for RADAR BARS (within selected leagues vs pooled) ---
league_col = "Competition_norm" if "Competition_norm" in df.columns else "Competition"
compute_within_league = st.checkbox("Percentiles within each league", value=True)

if compute_within_league and league_col in df.columns:
    percentile_df_chart = pd.DataFrame(index=df.index, columns=metrics, dtype=float)
    for m in metrics:
        percentile_df_chart[m] = (
            df.groupby(league_col, group_keys=False)[m]
              .apply(lambda s: pct_rank(s, lower_is_better=(m in LOWER_IS_BETTER)))
        )
else:
    percentile_df_chart = pd.DataFrame(index=df.index, columns=metrics, dtype=float)
    for m in metrics:
        percentile_df_chart[m] = pct_rank(df[m], lower_is_better=(m in LOWER_IS_BETTER))

percentile_df_chart = percentile_df_chart.round(1)

# --- B) Percentiles for 0–100 SCORE (baseline = WHOLE DATASET by position) ---
pos_col = "Six-Group Position"
if pos_col not in df_all.columns: df_all[pos_col] = np.nan
if pos_col not in df.columns:     df[pos_col]     = np.nan

percentile_df_globalpos_all = pd.DataFrame(index=df_all.index, columns=metrics, dtype=float)
for m in metrics:
    percentile_df_globalpos_all[m] = (
        df_all.groupby(pos_col, group_keys=False)[m]
              .apply(lambda s: pct_rank(s, lower_is_better=(m in LOWER_IS_BETTER)))
    )
percentile_df_globalpos = percentile_df_globalpos_all.loc[df.index, metrics].round(1)

# --- Assemble plot_data (radar uses the CHART percentiles) ---
metrics_df = df[metrics].copy()

keep_cols = [
    "Player", "Team within selected timeframe", "Team", "Age", "Height",
    "Positions played", "Minutes played", "Six-Group Position",
    "Competition", "Competition_norm", "Multiplier"
]
for c in keep_cols:
    if c not in df.columns: df[c] = np.nan

plot_data = pd.concat(
    [df[keep_cols], metrics_df, percentile_df_chart.add_suffix(" (percentile)")],
    axis=1
)

# ---------- Z + 0–100 score (based on WHOLE DATASET by position) ----------
sel_metrics = list(metric_groups.keys())
pct_for_score = percentile_df_globalpos[sel_metrics]
z_scores_all  = (pct_for_score - 50.0) / 15.0         # 50 -> 0, ~15 pct points = 1 SD
avg_z         = z_scores_all.mean(axis=1)

plot_data["Avg Z Score"]      = avg_z.values
plot_data["Multiplier"]        = plot_data["Multiplier"].fillna(1.0)
plot_data["Weighted Z Score"]  = plot_data["Avg Z Score"] * plot_data["Multiplier"]
plot_data["Score (0–100)"]     = (50.0 + 15.0 * plot_data["Weighted Z Score"]).clip(0, 100).round(1)

# Rank by the 0–100 score
plot_data["Rank"] = plot_data["Score (0–100)"].rank(ascending=False, method="min").astype(int)

# ---------- Chart ----------
def plot_radial_bar_grouped(player_name, plot_data, metric_groups, group_colors=None):
    import matplotlib.patches as mpatches

    if not isinstance(group_colors, dict) or len(group_colors) == 0:
        group_colors = {"Attacking": "crimson", "Possession": "seagreen", "Defensive": "royalblue", "Goalkeeping": "purple"}

    row = plot_data.loc[plot_data["Player"] == player_name]
    if row.empty:
        st.error(f"No player named '{player_name}' found.")
        return

    # For GK chart, keep Goalkeeping block together, then Possession
    group_order = ["Goalkeeping", "Possession", "Defensive", "Attacking", "Off The Ball"]
    sel_metrics = [m for g in group_order for m, gg in metric_groups.items() if gg == g]
    raw_vals = row[sel_metrics].values.flatten()
    pct_vals = row[[m + " (percentile)" for m in sel_metrics]].values.flatten()
    groups = [metric_groups[m] for m in sel_metrics]

    bar_colors = [group_colors.get(g, "grey") for g in groups]

    n = len(sel_metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)

    ax.bar(angles, pct_vals, width=2 * np.pi / n * 0.9, color=bar_colors, edgecolor=bar_colors, alpha=0.78)

    for ang, raw_val in zip(angles, raw_vals):
        try:
            txt = f"{float(raw_val):.2f}"
        except Exception:
            txt = "-"
        ax.text(ang, 50, txt, ha="center", va="center", color="black", fontsize=10, fontweight="bold")

    for i, ang in enumerate(angles):
        label = sel_metrics[i].replace(" per 90", "").replace(", %", " (%)")
        ax.text(ang, 108, label, ha="center", va="center", color="black", fontsize=10, fontweight="bold")

    present_groups = list(dict.fromkeys(groups))
    patches = [mpatches.Patch(color=group_colors.get(g, "grey"), label=g) for g in present_groups]
    if patches:
        fig.subplots_adjust(top=0.86, bottom=0.08)
        ax.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.06),
                  ncol=min(len(patches), 4), frameon=False)

    if "Weighted Z Score" in row.columns:
        weighted_z = float(row["Weighted Z Score"].values[0])
    else:
        z_scores = (pct_vals - 50) / 15
        avg_z = float(np.mean(z_scores))
        mult = float(row["Multiplier"].values[0]) if "Multiplier" in row.columns and pd.notnull(row["Multiplier"].values[0]) else 1.0
        weighted_z = avg_z * mult

    score_100 = None
    if "Score (0–100)" in row.columns and pd.notnull(row["Score (0–100)"].values[0]):
        score_100 = float(row["Score (0–100)"].values[0])

    age      = row["Age"].values[0] if "Age" in row else np.nan
    height   = row["Height"].values[0] if "Height" in row else np.nan
    team     = row["Team within selected timeframe"].values[0] if "Team within selected timeframe" in row else ""
    mins     = row["Minutes played"].values[0] if "Minutes played" in row else np.nan
    role     = row["Six-Group Position"].values[0] if "Six-Group Position" in row else ""
    rank_val = int(row["Rank"].values[0]) if "Rank" in row and pd.notnull(row["Rank"].values[0]) else None

    if "Competition_norm" in row.columns and pd.notnull(row["Competition_norm"].values[0]):
        comp = row["Competition_norm"].values[0]
    elif "Competition" in row.columns and pd.notnull(row["Competition"].values[0]):
        comp = row["Competition"].values[0]
    else:
        comp = ""

    top_parts = [player_name]
    if role: top_parts.append(role)
    if not pd.isnull(age):    top_parts.append(f"{int(age)} years old")
    if not pd.isnull(height): top_parts.append(f"{int(height)} cm")
    line1 = " | ".join(top_parts)

    bottom_parts = []
    if team:                 bottom_parts.append(team)
    if comp:                 bottom_parts.append(comp)
    if pd.notnull(mins):     bottom_parts.append(f"{int(mins)} mins")
    if rank_val is not None: bottom_parts.append(f"Rank #{rank_val}")

    if score_100 is not None:
        bottom_parts.append(f"Score {score_100:.0f}")
    else:
        bottom_parts.append(f"Z {weighted_z:.2f}")

    line2 = " | ".join(bottom_parts)

    ax.set_title(f"{line1}\n{line2}", color="black", size=22, pad=20, y=1.10)

    try:
        if logo is not None:
            imagebox = OffsetImage(np.array(logo), zoom=0.18)
            ab = AnnotationBbox(imagebox, (0, 0), frameon=False, box_alignment=(0.5, 0.5))
            ax.add_artist(ab)
    except Exception:
        pass

    st.pyplot(fig, use_container_width=True)

# ---------- Plot ----------
if st.session_state.selected_player:
    plot_radial_bar_grouped(
        st.session_state.selected_player,
        plot_data,
        metric_groups,
        group_colors
    )

# ---------- Ranking table ----------
st.markdown("### Players Ranked by Score (0–100)")

# Include Score (0–100) so we can sort by it and display it
cols_for_table = [
    "Player", "Positions played", "Competition_norm",
    "Score (0–100)", "Weighted Z Score", "Age", "Team", "Minutes played", "Rank"
]

# Ensure the columns exist (defensive)
for c in cols_for_table:
    if c not in plot_data.columns:
        plot_data[c] = np.nan

z_ranking = (
    plot_data[cols_for_table]
    .sort_values(by="Score (0–100)", ascending=False)
    .reset_index(drop=True)
)

# Nice display tweaks
z_ranking.rename(columns={"Competition_norm": "League"}, inplace=True)
z_ranking["Team"] = z_ranking["Team"].fillna("N/A")
if "Age" in z_ranking.columns:
    z_ranking["Age"] = z_ranking["Age"].apply(lambda x: int(x) if pd.notnull(x) else x)

# 1-based row index
z_ranking.index = np.arange(1, len(z_ranking) + 1)
z_ranking.index.name = "Row"

st.dataframe(z_ranking, use_container_width=True)
