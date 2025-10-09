import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from datetime import datetime
from auth import check_password
from branding import show_branding
import sqlite3
from pathlib import Path

# --- Database setup ---
DB_PATH = Path(__file__).parent / "favourites.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS favourites (
            player TEXT PRIMARY KEY,
            team TEXT,
            league TEXT,
            position TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()  # ensure DB exists on startup

# --- Ensure the radar page DB schema matches the favourites page ---
def migrate_favourites_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
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
    existing_cols = [r[1] for r in c.execute("PRAGMA table_info(favourites)").fetchall()]
    required_cols = {
        "colour": "TEXT DEFAULT ''",
        "comment": "TEXT DEFAULT ''",
        "visible": "INTEGER DEFAULT 1"
    }
    for col, dtype in required_cols.items():
        if col not in existing_cols:
            c.execute(f"ALTER TABLE favourites ADD COLUMN {col} {dtype}")
    conn.commit()
    conn.close()

migrate_favourites_db()

@st.cache_data(ttl=2, show_spinner=False)
def get_favourites_with_colours_live():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("SELECT player, colour, comment, visible FROM favourites")
        rows = c.fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    return {r[0]: {"colour": r[1], "comment": r[2], "visible": r[3]} for r in rows}

# --- Favourite helper functions ---
def add_favourite(player, team=None, league=None, position=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO favourites (player, team, league, position)
        VALUES (?, ?, ?, ?)
    """, (player, team, league, position))
    conn.commit()
    conn.close()

def remove_favourite(player):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM favourites WHERE player=?", (player,))
    conn.commit()
    conn.close()

def get_favourites():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT player, team, league, position FROM favourites ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return rows

# --- Password protection ---
from auth import check_password
from branding import show_branding

if not check_password():
    st.stop()

show_branding()
st.title("Player Radar")

APP_DIR = Path(__file__).parent          # pages/
ROOT_DIR = APP_DIR.parent                # repo root
ASSETS_DIR = ROOT_DIR / "assets"         # assets/ lives in root

def open_image(path: Path):
    try:
        return Image.open(path)
    except Exception:
        return None

# --- Basic password protection ---
PASSWORD = "Livi2025"

st.set_page_config(page_title="Livingston FC Recruitment App", layout="centered")

# ---------- Fixed group colours ----------
group_colors = {
    "Attacking":   "crimson",
    "Possession":  "seagreen",
    "Defensive":   "royalblue",
}

# ---------- Display name overrides for radar ----------
DISPLAY_NAMES = {
    "Player Season Fhalf Pressures 90": "Pressures in Opposition Half",
    "Deep Completions": "Completed Passes Final 1/3",
    "Turnovers": "Lost Balls",
    "Deep Progressions": "Progressions to Final 1/3",
    "Player Season Fhalf Ball Recoveries 90": "Ball Recovery Opp. Half",
    "Player Season Ball Recoveries 90": "Ball Recoveries",
}

# --- League name normalisation: StatsBomb -> your Opta names ---
LEAGUE_SYNONYMS = {
    "A-League": "Australia A-League Men",
    "2. Liga": "Austria 2. Liga",
    "Challenger Pro League": "Belgium Challenger Pro League",
    "First League": "Bulgaria First League",
    "1. HNL": "Croatia 1. HNL",
    "HNL": "Croatia 1. HNL",
    "Czech Liga": "Czech First Tier",
    "1st Division": "Denmark 1st Division",
    "Superliga": "Denmark Superliga",
    "Denmark Superliga": "Denmark Superliga",
    "League One": "England League One",
    "League Two": "England League Two",
    "National League": "England National League",
    "National League N / S": "England National League N/S",
    "Premium Liiga": "Estonia Premium Liiga",
    "Veikkausliiga": "Finland Veikkausliiga",
    "Championnat National": "France National 1",
    "Ligue 2": "Ligue 2",
    "France Ligue 2": "Ligue 2",
    "2. Bundesliga": "2. Bundesliga",
    "Germany 2. Bundesliga": "2. Bundesliga",
    "3. Liga": "Germany 3. Liga",
    "Super League": "Greece Super League 1",
    "Greece Super League": "Greece Super League 1",
    "NB I": "Hungary NB I",
    "Besta deild karla": "Iceland Besta Deild",
    "Serie C": "Italy Serie C",
    "J2 League": "Japan J2 League",
    "Virsliga": "Latvia Virsliga",
    "A Lyga": "Lithuania A Lyga",
    "Botola Pro": "Morocco Botola Pro",
    "Eredivisie": "Eredivisie",
    "Netherlands Eredivisie": "Eredivisie",
    "Eerste Divisie": "Netherlands Eerste Divisie",
    "1. Division": "Norway 1. Division",
    "Eliteserien": "Norway Eliteserien",
    "I Liga": "Poland 1 Liga",
    "Ekstraklasa": "Poland Ekstraklasa",
    "Segunda Liga": "Portugal Segunda Liga",
    "Liga Pro": "Portugal Segunda Liga",
    "Premier Division": "Republic of Ireland Premier Division",
    "Ireland Premier Division": "Republic of Ireland Premier Division",
    "Liga 1": "Romania Liga 1",
    "Championship": "Scotland Championship",
    "Premiership": "Scotland Premiership",
    "Super Liga": "Serbia Super Liga",
    "Slovakia Super Liga": "Slovakia 1. Liga",
    "Slovakia First League": "Slovakia 1. Liga",
    "1. Liga": "Slovakia 1. Liga",
    "1. Liga (SVN)": "Slovenia 1. Liga",
    "PSL": "South Africa Premier Division",
    "Allsvenskan": "Sweden Allsvenskan",
    "Superettan": "Sweden Superettan",
    "Challenge League": "Switzerland Challenge League",

    # --- Tunisia fixes ---
    "Ligue 1": "Tunisia Ligue 1",      # bare 'Ligue 1' should always mean Tunisia
    "Ligue 1 (TUN)": "Tunisia Ligue 1",
    "Tunisia Ligue 1": "Tunisia Ligue 1",
    "France Ligue 1": "Tunisia Ligue 1",

    # --- USA ---
    "USL Championship": "USA USL Championship",

    # --- Belgium top flight fixes ---
    "Jupiler Pro League": "Jupiler Pro League",
    "Belgium Pro League": "Jupiler Pro League",
    "Belgian Pro League": "Jupiler Pro League",
    "Belgium Jupiler Pro League": "Jupiler Pro League",
}

# ========== Role groups shown in filters ==========
SIX_GROUPS = [
    "Full Back", "Centre Back", "Number 6", "Number 8", "Winger", "Striker"
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
    # Full backs & wing backs
    "RIGHTBACK": "Full Back", "LEFTBACK": "Full Back",
    "RIGHTWINGBACK": "Full Back", "LEFTWINGBACK": "Full Back",
    # Centre backs
    "RIGHTCENTREBACK": "Centre Back", "LEFTCENTREBACK": "Centre Back", "CENTREBACK": "Centre Back",
    # Centre mid (generic) → duplicated into 6 & 8 later
    "CENTREMIDFIELDER": "Centre Midfield",
    "RIGHTCENTREMIDFIELDER": "Centre Midfield", "LEFTCENTREMIDFIELDER": "Centre Midfield",
    # Defensive mids → 6
    "DEFENSIVEMIDFIELDER": "Number 6", "RIGHTDEFENSIVEMIDFIELDER": "Number 6", "LEFTDEFENSIVEMIDFIELDER": "Number 6", "CENTREDEFENSIVEMIDFIELDER": "Number 6",
    # Attacking mids / 10 → 8
    "CENTREATTACKINGMIDFIELDER": "Number 8", "ATTACKINGMIDFIELDER": "Number 8", "RIGHTATTACKINGMIDFIELDER": "Number 8", "LEFTATTACKINGMIDFIELDER": "Number 8",
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
    return RAW_TO_SIX.get(tok, None)  # don’t force into Winger

# ========== Default template mapping ==========
DEFAULT_TEMPLATE = {
    "Full Back": "Full Back",
    "Centre Back": "Centre Back",
    "Number 6": "Number 6",
    "Number 8": "Number 8",
    "Winger": "Winger",
    "Striker": "Striker"
}

# ========== Radar metric sets ==========
position_metrics = {
    # ---------- Centre Back ----------
    "Centre Back": {
        "metrics": [
            # Attacking
            "NP Goals",
            # Possession
            "Passing%", "Pass OBV", "Pr. Long Balls", "UPr. Long Balls", "OBV", "Pr. Pass% Dif.",
            # Defensive
            "PAdj Interceptions", "PAdj Tackles", "Dribbles Stopped%",
            "Defensive Actions", "Aggressive Actions", "Fouls",
            "Aerial Wins", "Aerial Win%",
        ],
        "groups": {
            "PAdj Interceptions": "Defensive",
            "PAdj Tackles": "Defensive",
            "Dribbles Stopped%": "Defensive",
            "Defensive Actions": "Defensive",
            "Aggressive Actions": "Defensive",
            "Fouls": "Defensive",
            "Aerial Wins": "Defensive",
            "Aerial Win%": "Defensive",
            "Passing%": "Possession",
            "Pr. Pass% Dif.": "Possession",
            "Pr. Long Balls": "Possession",
            "UPr. Long Balls": "Possession",
            "OBV": "Possession",
            "Pass OBV": "Possession",
            "NP Goals": "Attacking",
        }
    },

    # ---------- Full Back ----------
    "Full Back": {
        "metrics": [
            "Passing%", "Pr. Pass% Dif.", "Successful Box Cross%", "Crossing%", "Deep Progressions",
            "Successful Dribbles", "Turnovers", "OBV", "Pass OBV",
            "Defensive Actions", "Aerial Win%", "PAdj Pressures",
            "PAdj Tack&Int", "Dribbles Stopped%", "Aggressive Actions", "Player Season Ball Recoveries 90"
        ],
        "groups": {
            "Passing%": "Possession",
            "Pr. Pass% Dif.": "Possession",
            "Successful Box Cross%": "Possession",
            "Crossing%": "Possession",
            "Deep Progressions": "Possession",
            "Successful Dribbles": "Possession",
            "Turnovers": "Possession",
            "OBV": "Possession",
            "Pass OBV": "Possession",
            "Defensive Actions": "Defensive",
            "Aerial Win%": "Defensive",
            "PAdj Pressures": "Defensive",
            "PAdj Tack&Int": "Defensive",
            "Dribbles Stopped%": "Defensive",
            "Aggressive Actions": "Defensive",
            "Player Season Ball Recoveries 90": "Defensive"
        }
    },

    # ---------- Number 6 ----------
    "Number 6": {
        "metrics": [
            # Attacking
            "xGBuildup", "xG Assisted",
            # Possession
            "Passing%", "Deep Progressions", "Turnovers", "OBV", "Pass OBV", "Pr. Pass% Dif.",
            # Defensive
            "PAdj Interceptions", "PAdj Tackles", "Dribbles Stopped%",
            "Aggressive Actions", "Aerial Win%", "Player Season Ball Recoveries 90", "Pressure Regains",
        ],
        "groups": {
            "Passing%": "Possession",
            "Deep Progressions": "Possession",
            "xGBuildup": "Attacking",
            "PAdj Interceptions": "Defensive",
            "PAdj Tackles": "Defensive",
            "Dribbles Stopped%": "Defensive",
            "Aggressive Actions": "Defensive",
            "Aerial Win%": "Defensive",
            "Player Season Ball Recoveries 90": "Defensive",
            "Pressure Regains": "Defensive",
            "Turnovers": "Possession",
            "OBV": "Possession",
            "Pass OBV": "Possession",
            "Pr. Pass% Dif.": "Possession",
            "xG Assisted": "Attacking",
        }
    },

    # ---------- Number 8 ----------
    "Number 8": {
        "metrics": [
            # Attacking
            "xGBuildup", "xG Assisted", "Shots", "xG", "NP Goals",
            # Possession
            "Passing%", "Deep Progressions", "OP Passes Into Box", "Pass OBV", "OBV", "Deep Completions",
            # Defensive
            "Pressure Regains", "PAdj Pressures", "Player Season Fhalf Ball Recoveries 90",
            "Aggressive Actions",
        ],
        "groups": {
            "Passing%": "Possession",
            "Deep Progressions": "Possession",
            "Deep Completions": "Possession",
            "xGBuildup": "Attacking",
            "xG Assisted": "Attacking",
            "OP Passes Into Box": "Possession",
            "Pass OBV": "Possession",
            "Shots": "Attacking",
            "xG": "Attacking",
            "NP Goals": "Attacking",
            "Pressure Regains": "Defensive",
            "PAdj Pressures": "Defensive",
            "Player Season Fhalf Ball Recoveries 90": "Defensive",
            "Aggressive Actions": "Defensive",
            "OBV": "Possession",
        }
    },

    # ---------- Winger ----------
    "Winger": {
        "metrics": [
            # Attacking
            "xG", "Shots", "xG/Shot", "Touches In Box", "OP xG Assisted", "NP Goals",
            # Possession
            "OP Passes Into Box", "Successful Box Cross%", "Passing%",
            "Successful Dribbles", "Turnovers", "OBV", "D&C OBV", "Fouls Won", "Deep Progressions",
            # Defensive
            "Player Season Fhalf Pressures 90",
        ],
        "groups": {
            "NP Goals": "Attacking",
            "xG": "Attacking",
            "Shots": "Attacking",
            "xG/Shot": "Attacking",
            "Touches In Box": "Attacking",
            "OP xG Assisted": "Attacking",
            "OP Passes Into Box": "Possession",
            "Successful Box Cross%": "Possession",
            "Passing%": "Possession",
            "Successful Dribbles": "Possession",
            "Fouls Won": "Possession",
            "Turnovers": "Possession",
            "OBV": "Possession",
            "D&C OBV": "Possession",
            "Player Season Fhalf Pressures 90": "Defensive",
        }
    },

    # ---------- Striker ----------
    "Striker": {
        "metrics": [
            "Aggressive Actions", "NP Goals", "xG", "Shots", "xG/Shot",
            "Goal Conversion%", 
            "Touches In Box", "xG Assisted",
            "Fouls Won", "Deep Completions", "OP Key Passes",
            "Aerial Win%", "Aerial Wins", "Player Season Fhalf Pressures 90",
        ],
        "groups": {
            "NP Goals": "Attacking",
            "xG": "Attacking",
            "Shots": "Attacking",
            "xG/Shot": "Attacking",
            "Goal Conversion%": "Attacking",
            "Touches In Box": "Attacking",
            "xG Assisted": "Attacking",
            "Fouls Won": "Possession",
            "Deep Completions": "Possession",
            "OP Key Passes": "Possession",
            "Aerial Win%": "Defensive",
            "Aerial Wins": "Defensive",
            "Aggressive Actions": "Defensive",
            "Player Season Fhalf Pressures 90": "Defensive"
        }
    }
}

# ---------- Data source: local repo ----------
DATA_PATH = ROOT_DIR / "statsbomb_player_stats_clean.csv"

# ---------- Helper functions for flexible loading ----------
def load_one_file(p: Path) -> pd.DataFrame:
    """Load a single CSV/XLSX file safely, handling encoding issues."""
    print(f"[DEBUG] Trying to load file at: {p.resolve()}")
    df = None

    if p.suffix.lower() in {".xlsx", ".xls"}:
        try:
            import openpyxl
            df = pd.read_excel(p, engine="openpyxl")
        except Exception as e:
            print(f"[DEBUG] Excel read failed: {e}, trying CSV fallback.")
    if df is None:
        for kwargs in [
            dict(sep=None, engine="python"),
            dict(),
            dict(encoding="latin1"),
        ]:
            try:
                df = pd.read_csv(p, **kwargs)
                break
            except Exception:
                continue
    if df is None:
        raise ValueError(f"Unsupported or unreadable file: {p.name}")
    print(f"[DEBUG] Loaded {p.name}, {len(df)} rows, {len(df.columns)} cols")
    return df


def _data_signature(path: Path):
    """Create a simple signature so Streamlit cache invalidates if file changes."""
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


def add_age_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add numeric Age column based on birth_date (if present)."""
    if "birth_date" not in df.columns:
        df["Age"] = np.nan
        return df

    today = datetime.today()
    df["Age"] = pd.to_datetime(df["birth_date"], errors="coerce").apply(
        lambda dob: today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        if pd.notna(dob) else np.nan
    )
    print(f"[DEBUG] Age column created. Non-null ages: {df['Age'].notna().sum()}")
    return df


# ---------- Preprocessing ----------
def preprocess_df(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    # --- Normalise Competition name ---
    if "Competition" in df.columns:
        df["Competition_norm"] = (
            df["Competition"].astype(str).str.strip().map(lambda x: LEAGUE_SYNONYMS.get(x, x))
        )
    else:
        df["Competition_norm"] = np.nan

# --- Merge league multipliers ---
try:
    multipliers_df = pd.read_excel(ROOT_DIR / "league_multipliers.xlsx")
    print("[DEBUG] Columns detected in league_multipliers.xlsx:", list(multipliers_df.columns))  # 👈 ADD THIS LINE

    if {"League", "Multiplier"}.issubset(multipliers_df.columns):
        df = df.merge(multipliers_df, left_on="Competition_norm", right_on="League", how="left")
        missing_mult = df[df["Multiplier"].isna()]["Competition_norm"].unique().tolist()
        if missing_mult:
            print(f"[DEBUG] Leagues without multipliers: {missing_mult}")
            st.warning(f"Some leagues did not match multipliers: {missing_mult}")
        df["Multiplier"] = df["Multiplier"].fillna(1.0)
    else:
        st.warning("league_multipliers.xlsx must have columns: 'League', 'Multiplier'. Using 1.0 for all.")
        df["Multiplier"] = 1.0
except Exception as e:
    print(f"[DEBUG] Failed to load multipliers: {e}")
    df["Multiplier"] = 1.0

    # --- Rename identifiers ---
    rename_map = {}
    if "Name" in df.columns:
        rename_map["Name"] = "Player"
    if "Primary Position" in df.columns:
        rename_map["Primary Position"] = "Position"
    if "Minutes" in df.columns:
        rename_map["Minutes"] = "Minutes played"

    # --- Metric renames / fixes ---
    rename_map.update({
        "Successful Box Cross %": "Successful Box Cross%",
        "Player Season Box Cross Ratio": "Successful Box Cross%",
        "Player Season Change In Passing Ratio": "Pr. Pass% Dif.",
        "Player Season Xgbuildup 90": "xGBuildup",
        "Player Season F3 Pressures 90": "Pressures in Final 1/3",
        "Player Season Pressured Long Balls 90": "Pr. Long Balls",
        "Player Season Unpressured Long Balls 90": "UPr. Long Balls",
    })
    df.rename(columns=rename_map, inplace=True)

    # --- Derive Successful Crosses ---
    cross_cols = [c for c in df.columns if "crosses" in c.lower()]
    crossperc_cols = [c for c in df.columns if "crossing%" in c.lower()]
    if cross_cols and crossperc_cols:
        c1 = cross_cols[0]
        c2 = crossperc_cols[0]
        df["Successful Crosses"] = (
            pd.to_numeric(df[c1], errors="coerce") *
            (pd.to_numeric(df[c2], errors="coerce") / 100.0)
        )
        print(f"[DEBUG] Created 'Successful Crosses' from {c1} and {c2}")
    else:
        print("[DEBUG] Could not find both 'Crosses' and 'Crossing%' columns.")

    # --- Derive Successful Dribbles ---
    if "Player Season Total Dribbles 90" in df.columns and "Player Season Failed Dribbles 90" in df.columns:
        df["Successful Dribbles"] = (
            pd.to_numeric(df["Player Season Total Dribbles 90"], errors="coerce").fillna(0)
            - pd.to_numeric(df["Player Season Failed Dribbles 90"], errors="coerce").fillna(0)
        )

    # --- Build "Positions played" ---
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

    # --- Fallbacks ---
    if "Team within selected timeframe" not in df.columns:
        df["Team within selected timeframe"] = df["Team"] if "Team" in df.columns else np.nan
    if "Height" not in df.columns:
        df["Height"] = np.nan

    # --- Six-Group Position mapping ---
    if "Position" in df.columns:
        df["Six-Group Position"] = df["Position"].apply(map_first_position_to_group)
    else:
        df["Six-Group Position"] = np.nan

    # --- Duplicate generic CMs into both 6 & 8 ---
    if "Six-Group Position" in df.columns:
        cm_mask = df["Six-Group Position"] == "Centre Midfield"
        if cm_mask.any():
            cm_rows = df.loc[cm_mask].copy()
            cm_as_6 = cm_rows.copy()
            cm_as_6["Six-Group Position"] = "Number 6"
            cm_as_8 = cm_rows.copy()
            cm_as_8["Six-Group Position"] = "Number 8"
            df = pd.concat([df, cm_as_6, cm_as_8], ignore_index=True)

    return df

# ---------- Cached Data Loader ----------
@st.cache_data(show_spinner=True)
def load_data_once(_sig=None):
    """Load and preprocess StatsBomb data once per session."""
    path = DATA_PATH
    sig = _data_signature(path)

    # Load one or many files
    if path.is_file():
        df_raw = load_one_file(path)
    else:
        files = sorted(
            f for f in path.iterdir()
            if f.is_file() and (f.suffix.lower() in {".csv", ".xlsx", ".xls"} or f.suffix == "")
        )
        if not files:
            raise FileNotFoundError(f"No data files found in {path.name}. Add CSV or XLSX.")
        frames = []
        for f in files:
            try:
                frames.append(load_one_file(f))
            except Exception as e:
                print(f"[WARNING] Skipping {f.name} ({e})")
        if not frames:
            raise ValueError("No readable files found in statsbombdata")
        df_raw = pd.concat(frames, ignore_index=True, sort=False)
        print(f"[DEBUG] Merged {len(files)} files, total rows {len(df_raw)}")

    df_raw = add_age_column(df_raw)
    df_preprocessed = preprocess_df(df_raw)
    print(f"[DEBUG] Data fully preprocessed. Rows: {len(df_preprocessed)}")

    return df_preprocessed
    
# ---------- Load & preprocess ----------
df_all_raw = load_data_once(_sig=_data_signature(DATA_PATH))

# ---------- Debug: list all 'cross' columns ----------
print("[DEBUG] Columns containing 'cross':")
for c in df_all_raw.columns:
    if "cross" in c.lower():
        print("   ", c)

# ---------- Clean raw column headers ----------
df_all_raw.columns = (
    df_all_raw.columns.astype(str)
    .str.strip()
    .str.replace(u"\xa0", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
)

# ---------- Add Age column from Birth Date ----------
if "Birth Date" in df_all_raw.columns:
    today = datetime.today()
    df_all_raw["Age"] = pd.to_datetime(df_all_raw["Birth Date"], errors="coerce").apply(
        lambda dob: today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        if pd.notna(dob) else np.nan
    )
    print(f"[DEBUG] Age column created. Non-null ages: {df_all_raw['Age'].notna().sum()}")

# ---------- Preprocess ----------
df_all = preprocess_df(df_all_raw)
print("[DEBUG] Final columns:", list(df_all.columns))

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

st.markdown("#### Choose league(s)")

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
    "Leagues",   # dummy label, won’t be shown
    options=all_leagues,
    default=st.session_state.league_selection,
    key="league_selection",
    label_visibility="collapsed"
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

# ---------- Minutes + Age filters (side by side) ----------
minutes_col = "Minutes played"
if minutes_col not in df.columns:
    df[minutes_col] = np.nan

c1, c2 = st.columns(2)

with c1:
    # Initialise in session_state if not already set
    if "min_minutes" not in st.session_state:
        st.session_state.min_minutes = 1000

    # Persist value in session_state
    st.session_state.min_minutes = st.number_input(
        "Minimum minutes to include",
        min_value=0,
        value=st.session_state.min_minutes,
        step=50,
        key="min_minutes_input"
    )

    min_minutes = st.session_state.min_minutes

    # Apply filter
    df["_minutes_numeric"] = pd.to_numeric(df[minutes_col], errors="coerce")
    df = df[df["_minutes_numeric"] >= min_minutes].copy()

    if df.empty:
        st.warning("No players meet the minutes threshold. Lower the minimum.")
        st.stop()


with c2:
    if "Age" in df.columns:
        df["_age_numeric"] = pd.to_numeric(df["Age"], errors="coerce")

        if df["_age_numeric"].notna().any():
            age_min = int(np.nanmin(df["_age_numeric"]))
            age_max = int(np.nanmax(df["_age_numeric"]))

            # Use session_state to persist age range
            if "age_range" not in st.session_state:
                st.session_state.age_range = (age_min, age_max)

            st.session_state.age_range = st.slider(
                "Age range to include",
                min_value=age_min,
                max_value=age_max,
                value=st.session_state.age_range,
                step=1,
                key="age_range_slider"
            )

            sel_min, sel_max = st.session_state.age_range
            df = df[df["_age_numeric"].between(sel_min, sel_max)].copy()

        else:
            st.info("Age column has no numeric values, age filter skipped.")
    else:
        st.info("No Age column found, age filter skipped.")

st.caption(f"Filtering on '{minutes_col}' ≥ {min_minutes}. Players remaining: {len(df)}")

# ---------- Position Group Section ----------
st.markdown("#### 🟡 Select Position Group")

# Build list of available groups from data
available_groups = []
if "Six-Group Position" in df.columns:
    available_groups = [g for g in SIX_GROUPS if g in df["Six-Group Position"].unique()]

# Use session_state to persist the last selection
if "selected_groups" not in st.session_state:
    st.session_state.selected_groups = []

selected_groups = st.multiselect(
    "Position Groups",
    options=available_groups,
    default=st.session_state.selected_groups,
    label_visibility="collapsed",
    key="pos_group_multiselect"
)

# Apply filter if groups selected
if selected_groups:
    df = df[df["Six-Group Position"].isin(selected_groups)].copy()
    if df.empty:
        st.warning("No players after group filter. Clear filters or choose different groups.")
        st.stop()

# Track currently selected group if only one is chosen
current_single_group = selected_groups[0] if len(selected_groups) == 1 else None

# ---------- Session state setup ----------
if "template_select" not in st.session_state:
    st.session_state.template_select = list(position_metrics.keys())[0]
if "last_template_choice" not in st.session_state:
    st.session_state.last_template_choice = st.session_state.template_select
if "manual_override" not in st.session_state:
    st.session_state.manual_override = False
if "last_groups_tuple" not in st.session_state:
    st.session_state.last_groups_tuple = tuple()

# ---------- Auto-sync template with group ----------
if tuple(selected_groups) != st.session_state.last_groups_tuple:
    if len(selected_groups) == 1:
        pos = selected_groups[0]
        if pos in position_metrics:
            st.session_state.template_select = pos
            st.session_state.manual_override = False
    st.session_state.last_groups_tuple = tuple(selected_groups)

# ---------- Template Section ----------
st.markdown("#### 📊 Choose Radar Template")

template_names = list(position_metrics.keys())
if st.session_state.template_select not in template_names:
    st.session_state.template_select = template_names[0]

selected_position_template = st.selectbox(
    "Radar Template",   # dummy label (required)
    template_names,
    index=template_names.index(st.session_state.template_select),
    key="template_select",
    label_visibility="collapsed"   # hides the text
)

# Handle manual override
if st.session_state.template_select != st.session_state.last_template_choice:
    st.session_state.manual_override = True
    st.session_state.last_template_choice = st.session_state.template_select

# ---------- Build metric pool for Essential Criteria ----------
current_template_name = st.session_state.template_select or list(position_metrics.keys())[0]
current_metrics = position_metrics[current_template_name]["metrics"]

for m in current_metrics:
    if m not in df.columns:
        df[m] = 0
df[current_metrics] = df[current_metrics].fillna(0)

# ---------- Session state setup for Essential Criteria ----------
if "ec_rows" not in st.session_state:
    st.session_state.ec_rows = 1

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

# --- Initialise session state if missing ---
if "selected_player" not in st.session_state:
    st.session_state.selected_player = None

if st.session_state.selected_player not in players:
    st.session_state.selected_player = players[0]

selected_player = st.selectbox(
    "Choose a player",
    players,
    index=players.index(st.session_state.selected_player) if st.session_state.selected_player in players else 0,
    key="player_select"
)
st.session_state.selected_player = selected_player

# ---------- Metrics + percentiles ----------
metrics = position_metrics[selected_position_template]["metrics"]
metric_groups = position_metrics[selected_position_template]["groups"]

# Ensure columns exist and are numeric in both df_all and df
for m in metrics:
    if m not in df_all.columns:
        df_all[m] = 0
    if m not in df.columns:
        df[m] = 0
    df_all[m] = pd.to_numeric(df_all[m], errors="coerce").fillna(0)
    df[m] = pd.to_numeric(df[m], errors="coerce").fillna(0)

# Metrics where lower values are better (raw values unchanged; only percentiles invert)
LOWER_IS_BETTER = {
    "Turnovers",
    "Fouls",
    "Pr. Long Balls",
    "UPr. Long Balls",
}

def pct_rank(series: pd.Series, lower_is_better: bool) -> pd.Series:
    # Convert series to numeric, handle invalid values
    series = pd.to_numeric(series, errors="coerce").fillna(0)
    # pandas: rank(pct=True, ascending=True) -> smallest ≈ 0, largest = 1.0
    r = series.rank(pct=True, ascending=True)
    if lower_is_better:
        p = 1.0 - r  # smaller raw -> larger percentile
    else:
        p = r  # larger raw -> larger percentile
    return (p * 100.0).round(1)

# --- A) Percentiles for RADAR BARS (within selected leagues vs pooled) ---
league_col = "Competition_norm" if "Competition_norm" in df.columns else "Competition"
compute_within_league = st.checkbox("Percentiles within each league", value=True, key="percentiles_within_league")
percentile_df_chart = pd.DataFrame(index=df.index, columns=metrics, dtype=float)
if compute_within_league and league_col in df.columns:
    for m in metrics:
        try:
            percentile_df_chart[m] = (
                df.groupby(league_col, group_keys=False)[m]
                  .apply(lambda s: pct_rank(s, lower_is_better=(m in LOWER_IS_BETTER)))
            )
        except Exception as e:
            print(f"[DEBUG] Percentile calc failed for {m}: {e}")
            percentile_df_chart[m] = 50.0  # Neutral percentile if calculation fails
else:
    for m in metrics:
        try:
            percentile_df_chart[m] = pct_rank(df[m], lower_is_better=(m in LOWER_IS_BETTER))
        except Exception as e:
            print(f"[DEBUG] Percentile calc failed for {m}: {e}")
            percentile_df_chart[m] = 50.0  # Neutral percentile if calculation fails
percentile_df_chart = percentile_df_chart.fillna(50.0).round(1)  # Fill any remaining NaN with neutral percentile

# --- B) Percentiles for 0–100 SCORE (baseline = WHOLE DATASET by position) ---
pos_col = "Six-Group Position"
if pos_col not in df_all.columns: df_all[pos_col] = np.nan
if pos_col not in df.columns: df[pos_col] = np.nan
percentile_df_globalpos_all = pd.DataFrame(index=df_all.index, columns=metrics, dtype=float)
for m in metrics:
    try:
        percentile_df_globalpos_all[m] = (
            df_all.groupby(pos_col, group_keys=False)[m]
                  .apply(lambda s: pct_rank(s, lower_is_better=(m in LOWER_IS_BETTER)))
        )
    except Exception as e:
        print(f"[DEBUG] Global percentile calc failed for {m}: {e}")
        percentile_df_globalpos_all[m] = 50.0  # Neutral percentile
percentile_df_globalpos = percentile_df_globalpos_all.loc[df.index, metrics].fillna(50.0).round(1)

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

# ---------- Z + 0–100 score (raw Z-scores for ranking, percentiles for radar only) ----------

# Metrics for scoring (same as chart)
sel_metrics = list(metric_groups.keys())

# --- A) Percentiles for SCORE BASELINE (full dataset by position, for reference) ---
pos_col = "Six-Group Position"
if pos_col not in df_all.columns: df_all[pos_col] = np.nan
if pos_col not in df.columns: df[pos_col] = np.nan

percentile_df_globalpos_all = pd.DataFrame(index=df_all.index, columns=sel_metrics, dtype=float)
for m in sel_metrics:
    # Ensure metric is numeric
    df_all[m] = pd.to_numeric(df_all[m], errors="coerce").fillna(0)
    percentile_df_globalpos_all[m] = (
        df_all.groupby(pos_col, group_keys=False)[m]
              .apply(lambda s: pct_rank(s, lower_is_better=(m in LOWER_IS_BETTER)))
    )
percentile_df_globalpos = percentile_df_globalpos_all.loc[df.index, sel_metrics].round(1)

# --- B) Raw Z-Scores for RANKING (full dataset by position, manual calc) ---
# Compute Z per metric: (raw - mean)/std per position; invert lower-better
raw_z_all = pd.DataFrame(index=df_all.index, columns=sel_metrics, dtype=float)
for m in sel_metrics:
    # Ensure metric is numeric
    df_all[m] = pd.to_numeric(df_all[m], errors="coerce").fillna(0)
    # Manual Z-score: (x - mean)/std per position group
    group_stats = df_all.groupby(pos_col)[m].agg(['mean', 'std']).fillna(0)
    z_per_group = df_all.groupby(pos_col)[m].transform(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0)
    if m in LOWER_IS_BETTER:
        z_per_group *= -1  # Invert: lower raw → higher Z
    raw_z_all[m] = z_per_group.fillna(0)  # Fill NaN Z as 0 (neutral)

avg_z_all = raw_z_all.mean(axis=1)  # Average Z across metrics

# Apply to full df_all
df_all["Avg Z Score"] = pd.to_numeric(avg_z_all, errors="coerce").fillna(0)
df_all["Multiplier"] = pd.to_numeric(df_all.get("Multiplier", 1.0), errors="coerce").fillna(1.0)
df_all["Weighted Z Score"] = df_all["Avg Z Score"] * df_all["Multiplier"]

print("[DEBUG] Multiplier stats:")
print("   Unique multipliers found:", sorted(df_all["Multiplier"].dropna().unique())[:15])
print("   Weighted Z Score range:", df_all["Weighted Z Score"].min(), "→", df_all["Weighted Z Score"].max())

# Show a few sample rows to confirm multiplier usage
print(df_all[["Competition_norm", "Multiplier", "Avg Z Score", "Weighted Z Score"]].head(10).to_string())

# For current view (plot_data): subset the raw Z's and avg
raw_z_view = raw_z_all.loc[df.index, sel_metrics]
avg_z_view = raw_z_view.mean(axis=1)
plot_data["Avg Z Score"] = pd.to_numeric(avg_z_view, errors="coerce").fillna(0)
plot_data["Multiplier"] = pd.to_numeric(plot_data["Multiplier"], errors="coerce").fillna(1.0)
plot_data["Weighted Z Score"] = plot_data["Avg Z Score"] * plot_data["Multiplier"]

# Flag eligibility
plot_data["_mins_numeric"] = pd.to_numeric(plot_data["Minutes played"], errors="coerce")
plot_data["Eligible Mins?"] = plot_data["_mins_numeric"] >= 600

# 3) Anchors from eligible (>=600 mins) on weighted Z (full dataset)
anchor_minutes_floor = 600
user_min_minutes = max(anchor_minutes_floor, min_minutes)
_mins_all = pd.to_numeric(df_all.get("Minutes played", np.nan), errors="coerce")
eligible = df_all[_mins_all >= user_min_minutes].copy()
if eligible.empty:
    st.warning(f"No players with >= {user_min_minutes} mins for anchors. Falling back to full dataset.")
    eligible = df_all.copy()

anchor_minmax = (
    eligible.groupby(pos_col)["Weighted Z Score"]
            .agg(_scale_min="min", _scale_max="max")
            .fillna(0)
)

# Warn on small pools
eligible_counts = eligible.groupby(pos_col).size()
small_positions = eligible_counts[eligible_counts < 5].index.tolist()
if small_positions:
    st.warning(f"Small eligible pools (<5 players) for anchors in positions: {', '.join(small_positions)}. Scores may bunch up.")

# 4) Scale to 0-100 for all in plot_data
plot_data = plot_data.merge(anchor_minmax, left_on=pos_col, right_index=True, how="left")

def _minmax_score(val, lo, hi):
    # Convert inputs to float, handle non-numeric cases
    try:
        val = float(val)
    except (TypeError, ValueError):
        val = 0.0
    try:
        lo = float(lo)
    except (TypeError, ValueError):
        lo = 0.0
    try:
        hi = float(hi)
    except (TypeError, ValueError):
        hi = 1.0
    if pd.isna(val) or pd.isna(lo) or pd.isna(hi):
        return 0.0
    if not np.isfinite(val):
        val = 0.0
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi):
        hi = 1.0
    if hi <= lo:
        return 50.0
    return float(np.clip((val - lo) / (hi - lo) * 100.0, 0.0, 100.0))

plot_data["Score (0–100)"] = [
    _minmax_score(v, lo, hi)
    for v, lo, hi in zip(plot_data["Weighted Z Score"], plot_data["_scale_min"], plot_data["_scale_max"])
]
plot_data["Score (0–100)"] = pd.to_numeric(plot_data["Score (0–100)"], errors="coerce").round(1).fillna(0)

# 5) Rank all filtered players
plot_data["Rank"] = plot_data["Score (0–100)"].rank(ascending=False, method="min").astype(int)

# Clean up
plot_data.drop(columns=["_scale_min", "_scale_max", "_mins_numeric"], inplace=True, errors="ignore")

# Debug
print("[DEBUG] Anchor minutes floor =", user_min_minutes)
print("[DEBUG] Eligible counts per pos:", dict(eligible_counts))
print("[DEBUG] Sample anchor ranges:",
      anchor_minmax.reset_index()
                   .rename(columns={pos_col: "Pos"})
                   .head(6)
                   .to_dict(orient="records"))
print("[DEBUG] Sample Weighted Z Scores:", plot_data[["Player", "Weighted Z Score"]].head().to_dict())
print("[DEBUG] Sample Score (0-100):", plot_data[["Player", "Score (0–100)"]].head().to_dict())
# ---------- Chart ----------
def plot_radial_bar_grouped(player_name, plot_data, metric_groups, group_colors=None):
    import matplotlib.patches as mpatches
    from matplotlib import colormaps as mcm

    if not isinstance(group_colors, dict) or len(group_colors) == 0:
        group_colors = {
            "Attacking": "crimson",
            "Possession": "seagreen",
            "Defensive": "royalblue",
        }

    # Get player row safely
    row_df = plot_data.loc[plot_data["Player"] == player_name]
    if row_df.empty:
        st.error(f"No player named '{player_name}' found.")
        return

    row = row_df.iloc[0]

    # Get all metric names (with valid percentile cols)
    group_order = ["Possession", "Defensive", "Attacking", "Off The Ball"]
    ordered_metrics = [m for g in group_order for m, gg in metric_groups.items() if gg == g]

    valid_metrics, valid_pcts = [], []
    for m in ordered_metrics:
        pct_col = f"{m} (percentile)"
        if m in row.index and pct_col in row.index:
            valid_metrics.append(m)
            valid_pcts.append(pct_col)

    if not valid_metrics:
        st.warning("No valid metrics available to plot for this player.")
        return

    # Retrieve numeric values safely
    raw_vals = pd.to_numeric(row[valid_metrics], errors="coerce").fillna(0).to_numpy()
    pct_vals = pd.to_numeric(row[valid_pcts], errors="coerce").fillna(50).to_numpy()  # default neutral 50

    n = len(valid_metrics)
    if n == 0:
        st.warning("No valid numeric metrics found for this player.")
        return

    # Groups and colours
    groups = [metric_groups.get(m, "Unknown") for m in valid_metrics]
    cmap = mcm.get_cmap("RdYlGn")
    norm = mcolors.Normalize(vmin=0, vmax=100)
    bar_colors = [cmap(norm(v)) for v in pct_vals]

    # Angles for radar
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # --- Plot setup ---
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)

    # --- Bars ---
    ax.bar(
        angles, pct_vals,
        width=2 * np.pi / n * 0.85,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9
    )

    # Raw values inside the chart
    for ang, raw_val in zip(angles, raw_vals):
        txt = f"{raw_val:.2f}" if np.isfinite(raw_val) else "-"
        ax.text(ang, 50, txt, ha="center", va="center", color="black", fontsize=10, fontweight="bold")

    # Metric labels
    for ang, m in zip(angles, valid_metrics):
        label = DISPLAY_NAMES.get(m, m)
        label = label.replace(" per 90", "").replace(", %", " (%)")
        color = group_colors.get(metric_groups.get(m, "Unknown"), "black")
        ax.text(ang, 108, label, ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    # --- Legend ---
    present_groups = list(dict.fromkeys(groups))
    patches = [mpatches.Patch(color=group_colors.get(g, "grey"), label=g) for g in present_groups]
    if patches:
        fig.subplots_adjust(top=0.86, bottom=0.08)
        ax.legend(
            handles=patches,
            loc="upper center", bbox_to_anchor=(0.5, -0.06),
            ncol=min(len(patches), 4), frameon=False
        )

    # --- Player Info ---
    weighted_z = float(row.get("Weighted Z Score", 0) or 0)
    score_100 = row.get("Score (0–100)")
    score_100 = float(score_100) if pd.notnull(score_100) else None

    age = row.get("Age", np.nan)
    height = row.get("Height", np.nan)
    team = row.get("Team within selected timeframe", "") or ""
    mins = row.get("Minutes played", np.nan)
    role = row.get("Six-Group Position", "") or ""
    rank_v = int(row.get("Rank", 0)) if pd.notnull(row.get("Rank", 0)) else None

    comp = row.get("Competition_norm") or row.get("Competition") or ""

    # Title text
    top_parts = [player_name]
    if role: top_parts.append(role)
    if pd.notnull(age): top_parts.append(f"{int(age)} years old")
    if pd.notnull(height): top_parts.append(f"{int(height)} cm")
    line1 = " | ".join(top_parts)

    bottom_parts = []
    if team: bottom_parts.append(team)
    if comp: bottom_parts.append(comp)
    if pd.notnull(mins): bottom_parts.append(f"{int(mins)} mins")
    if rank_v: bottom_parts.append(f"Rank #{rank_v}")
    if score_100 is not None:
        bottom_parts.append(f"{score_100:.0f}/100")
    else:
        bottom_parts.append(f"Z {weighted_z:.2f}")
    line2 = " | ".join(bottom_parts)

    ax.set_title(f"{line1}\n{line2}", color="black", size=22, pad=20, y=1.10)

    # Logo overlay if available
    try:
        if "logo" in globals() and logo is not None:
            imagebox = OffsetImage(np.array(logo), zoom=0.18)
            ab = AnnotationBbox(imagebox, (0, 0), frameon=False, box_alignment=(0.5, 0.5))
            ax.add_artist(ab)
    except Exception:
        pass

    st.pyplot(fig, width="stretch")

# ---------- Plot ----------
if st.session_state.selected_player:
    plot_radial_bar_grouped(
        st.session_state.selected_player,
        plot_data,
        metric_groups,
        group_colors
    )

# ---------- Ranking table with favourites ----------
st.markdown("### Players Ranked by Score (0–100)")

# Include key columns
cols_for_table = [
    "Player", "Positions played", "Team", "Competition_norm", "Multiplier",
    "Score (0–100)", "Age", "Minutes played", "Rank"
]

for c in cols_for_table:
    if c not in plot_data.columns:
        plot_data[c] = np.nan

z_ranking = plot_data[cols_for_table].copy()

# Clean up columns
z_ranking.rename(columns={"Competition_norm": "League"}, inplace=True)
z_ranking["Team"] = z_ranking["Team"].fillna("N/A")

if "Age" in z_ranking.columns:
    z_ranking["Age"] = z_ranking["Age"].apply(lambda x: int(x) if pd.notnull(x) else x)

z_ranking["Minutes played"] = pd.to_numeric(z_ranking["Minutes played"], errors="coerce").fillna(0).astype(int)
z_ranking["Multiplier"] = pd.to_numeric(z_ranking["Multiplier"], errors="coerce").fillna(1.0).round(3)

# Deduplicate and rank
z_ranking = (
    z_ranking.sort_values("Score (0–100)", ascending=False)
             .groupby("Player", as_index=False)
             .first()
)
z_ranking["Rank"] = z_ranking["Score (0–100)"].rank(ascending=False, method="min").astype(int)
z_ranking = z_ranking.sort_values("Rank", ascending=True).reset_index(drop=True)
z_ranking.index = np.arange(1, len(z_ranking) + 1)
z_ranking.index.name = "Row"

# ============================================================
# 🔄 FAVOURITES SYNC (Integrated)
# ============================================================
def get_favourites_with_colours():
    """Fetch all favourites and their colour/comment/visibility."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("SELECT player, colour, comment, visible FROM favourites")
        rows = c.fetchall()
    except sqlite3.OperationalError:
        rows = []  # handle case if columns not yet added
    conn.close()
    return {r[0]: {"colour": r[1], "comment": r[2], "visible": r[3]} for r in rows}

def upsert_favourite(player, team, league, position, colour="🟡 Yellow", comment="", visible=1):
    """Add or update a favourite record."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO favourites (player, team, league, position, colour, comment, visible, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(player) DO UPDATE SET
            team=excluded.team,
            league=excluded.league,
            position=excluded.position,
            colour=excluded.colour,
            comment=excluded.comment,
            visible=excluded.visible,
            timestamp=CURRENT_TIMESTAMP
    """, (player, team, league, position, colour, comment, visible))
    conn.commit()
    conn.close()

def hide_favourite(player):
    """Soft-remove favourite (visible = 0 instead of delete)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE favourites SET visible=0, timestamp=CURRENT_TIMESTAMP WHERE player=?", (player,))
    conn.commit()
    conn.close()

# ============================================================
# 🟢 LOAD FAVOURITES AND APPLY COLOURS
# ============================================================
favs = get_favourites_with_colours_live()

# Map all shapes of the four statuses to their emoji
COLOUR_EMOJI = {
    "🟣 Needs Checked": "🟣",
    "🟡 Monitor": "🟡",
    "🟢 Go": "🟢",
    "🔴 No Further Interest": "🔴",
    "Needs Checked": "🟣",
    "Monitor": "🟡",
    "Go": "🟢",
    "No Further Interest": "🔴",
    "🟣": "🟣",
    "🟡": "🟡",
    "🟢": "🟢",
    "🔴": "🔴",
}

def colourize_player_name(name: str) -> str:
    """Attach the correct emoji to the player name, even if hidden."""
    data = favs.get(name)
    if not data:
        return name
    colour = str(data.get("colour", "")).strip()
    emoji = COLOUR_EMOJI.get(colour, "")
    return f"{emoji} {name}" if emoji else name

# ============================================================
# 🧾 ENSURE TABLE COLUMNS EXIST & REORDER
# ============================================================
z_ranking["Player (coloured)"] = z_ranking["Player"].apply(colourize_player_name)
z_ranking["⭐ Favourite"] = z_ranking["Player"].apply(
    lambda n: bool(favs.get(n, {}).get("visible", 0))
)

required_cols = [
    "⭐ Favourite", "Player (coloured)", "Positions played", "Team", "League",
    "Multiplier", "Score (0–100)", "Age", "Minutes played", "Rank"
]
for col in required_cols:
    if col not in z_ranking.columns:
        z_ranking[col] = np.nan

z_ranking = z_ranking[required_cols]

# ============================================================
# 📋 EDITABLE TABLE
# ============================================================
edited_df = st.data_editor(
    z_ranking,
    column_config={
        "Player (coloured)": st.column_config.TextColumn(
            "Player",
            help="Shows Favourites colour (🟢🟡🔴🟣 only if marked)"
        ),
        "⭐ Favourite": st.column_config.CheckboxColumn(
            "⭐ Favourite",
            help="Mark or unmark as favourite (auto-syncs with Favourites page)"
        ),
        "Multiplier": st.column_config.NumberColumn(
            "League Weight",
            help="League weighting applied in ranking",
            format="%.3f"
        ),
    },
    hide_index=False,
    width="stretch",
    key=f"ranking_editor_{selected_position_template}",  # ✅ unique key avoids all duplication
)

# ============================================================
# 💾 APPLY CHANGES TO favourites.db
# ============================================================
for _, row in edited_df.iterrows():
    player_raw = str(row.get("Player (coloured)", "")).strip()
    player_name = re.sub(r"^[🟢🟡🔴🟣]\s*", "", player_raw).strip()

    team = row.get("Team", "")
    league = row.get("League", "")
    position = row.get("Positions played", "")
    is_fav = bool(row.get("⭐ Favourite", False))

    current_data = favs.get(player_name, {})
    colour = current_data.get("colour", "")
    comment = current_data.get("comment", "")

    if is_fav:
        upsert_favourite(player_name, team, league, position, colour=colour, comment=comment, visible=1)
    else:
        hide_favourite(player_name)
