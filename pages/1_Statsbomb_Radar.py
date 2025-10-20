import random
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
from supabase import create_client
from lib.favourites_repo import upsert_favourite, hide_favourite, list_favourites
from datetime import datetime, timezone
from openai import OpenAI
client = OpenAI(api_key=st.secrets["OpenAI"]["OPENAI_API_KEY"])

# ========= DEBUG MARKERS =========
print("[DEBUG_LOOP] ---- PAGE START ----")
print("[DEBUG] Run marker:", random.randint(1000, 9999))
print("[DEBUG] Password OK?", st.session_state.get("password_ok"))

# ========= PAGE CONFIG =========
st.set_page_config(page_title="Livingston FC Recruitment App", layout="centered")

# ========= AUTH / BRANDING =========
from auth import check_password
from branding import show_branding

# Password gate (single controlled rerun)
if not check_password():
    st.stop()

show_branding()
st.title("Statsbomb Radar")

# ========= PATHS =========
APP_DIR = Path(__file__).parent          # pages/
ROOT_DIR = APP_DIR.parent                # repo root
ASSETS_DIR = ROOT_DIR / "assets"         # assets/ lives in root


def open_image(path: Path):
    """Safe image loader."""
    try:
        return Image.open(path)
    except Exception:
        return None


# ========= FIXED GROUP COLOURS =========
group_colors = {
    "Attacking": "crimson",
    "Possession": "seagreen",
    "Defensive": "royalblue",
}

# ========= DISPLAY NAME OVERRIDES =========
DISPLAY_NAMES = {
    "Player Season Fhalf Pressures 90": "Pressures in Opposition Half",
    "Deep Completions": "Completed Passes Final 1/3",
    "Turnovers": "Lost Balls",
    "Deep Progressions": "Progressions to Final 1/3",
    "Player Season Fhalf Ball Recoveries 90": "Ball Recovery Opp. Half",
    "Player Season Ball Recoveries 90": "Ball Recoveries",
}

# ========= LEAGUE SYNONYMS =========
LEAGUE_SYNONYMS = {
    # --- Australia ---
    "A-League": "Australia A-League Men",
    "Australia A-League": "Australia A-League Men",

    # --- Austria ---
    "2. Liga": "Austria 2. Liga",

    # --- Belgium ---
    "Challenger Pro League": "Belgium Challenger Pro League",

    # --- Bulgaria ---
    "First League": "Bulgaria First League",

    # --- Croatia ---
    "1. HNL": "Croatia 1. HNL",

    # --- Czech Republic ---
    "Czech Liga": "Czech First Tier",

    # --- Denmark ---
    "1st Division": "Denmark 1st Division",
    "Superliga": "Denmark Superliga",

    # --- England ---
    "League One": "England League One",
    "League Two": "England League Two",
    "National League": "England National League",

    # --- Estonia ---
    "Premium Liiga": "Estonia Premium Liiga",

    # --- Finland ---
    "Veikkausliiga": "Finland Veikkausliiga",

    # --- France ---
    "Championnat National": "France National 1",
    "Ligue 2": "Ligue 2",

    # --- Germany ---
    "3. Liga": "Germany 3. Liga",

    # --- Greece ---
    "Super League": "Greece Super League 1",

    # --- Hungary ---
    "NB I": "Hungary NB I",

    # --- Iceland ---
    "Besta deild karla": "Iceland Besta Deild",

    # --- Italy ---
    "Serie C": "Italy Serie C",

    # --- Japan ---
    "J2 League": "Japan J2 League",

    # --- Latvia ---
    "Virsliga": "Latvia Virsliga",

    # --- Lithuania ---
    "A Lyga": "Lithuania A Lyga",

    # --- Morocco ---
    "Botola Pro": "Morocco Botola Pro",

    # --- Netherlands ---
    "Eredivisie": "Eredivisie",
    "Eerste Divisie": "Netherlands Eerste Divisie",

    # --- Norway ---
    "1. Division": "Norway 1. Division",
    "Eliteserien": "Norway Eliteserien",

    # --- Poland ---
    "I Liga": "Poland 1 Liga",
    "Ekstraklasa": "Poland Ekstraklasa",

    # --- Portugal ---
    "Segunda Liga": "Portugal Segunda Liga",

    # --- Ireland ---
    "Premier Division": "Republic of Ireland Premier Division",

    # --- Romania ---
    "Liga 1": "Romania Liga 1",

    # --- Scotland ---
    "Championship": "Scotland Championship",
    "Premiership": "Scotland Premiership",

    # --- Serbia ---
    "Super Liga": "Serbia Super Liga",

    # --- Slovenia ---
    "1. Liga (SVN)": "Slovenia 1. Liga",

    # --- South Africa ---
    "PSL": "South Africa Premier Division",

    # --- Sweden ---
    "Allsvenskan": "Sweden Allsvenskan",
    "Superettan": "Sweden Superettan",

    # --- Switzerland ---
    "Challenge League": "Switzerland Challenge League",

    # --- Tunisia ---
    "Tunisia Ligue 1": "Tunisia Ligue 1",

    # --- USA ---
    "USL Championship": "USA USL Championship",

    # --- Belgium alt ---
    "Jupiler Pro League": "Jupiler Pro League",
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


def preprocess_df(df_in: pd.DataFrame) -> pd.DataFrame:
    """Clean, normalise, and enrich the StatsBomb player dataset."""
    df = df_in.copy()

    # ============================================================
    # 🏆 1. Normalise Competition names
    # ============================================================
    if "Competition" in df.columns:
        df["Competition_norm"] = (
            df["Competition"].astype(str).str.strip().map(lambda x: LEAGUE_SYNONYMS.get(x, x))
        )
    else:
        df["Competition_norm"] = np.nan

    # ============================================================
    # ⚖️ 2. Merge League Multipliers (using Competition_ID first)
    # ============================================================
    try:
        multipliers_path = ROOT_DIR / "league_multipliers.xlsx"
        m = pd.read_excel(multipliers_path)

        # Normalise Excel headers
        m.columns = m.columns.str.strip().str.lower()  # -> competition_id, league, multiplier
        df.columns = df.columns.str.strip()  # ensure consistency

        merged = False

        if "competition_id" in df.columns and "competition_id" in m.columns:
            df = df.merge(m, on="competition_id", how="left")
            print("[DEBUG] ✅ Merged league multipliers using Competition_ID")
            merged = True
        elif "Competition_norm" in df.columns and "league" in m.columns:
            df = df.merge(m, left_on="Competition_norm", right_on="league", how="left")
            print("[DEBUG] ✅ Merged league multipliers using Competition_norm")
            merged = True
        elif "Competition" in df.columns and "league" in m.columns:
            df = df.merge(m, left_on="Competition", right_on="league", how="left")
            print("[DEBUG] ⚠️ Fallback merge on raw Competition name")
            merged = True
        else:
            print("[DEBUG] ⚠️ No matching key found for league multipliers; using 1.0 for all.")
            df["multiplier"] = 1.0

        # Safely assign numeric multiplier
        if "multiplier" in df.columns:
            df["Multiplier"] = pd.to_numeric(df["multiplier"], errors="coerce").fillna(1.0)
        elif "Multiplier" in df.columns:
            df["Multiplier"] = pd.to_numeric(df["Multiplier"], errors="coerce").fillna(1.0)
        else:
            df["Multiplier"] = 1.0

        # Debug output
        print("[DEBUG] Unique multipliers after merge:", sorted(df["Multiplier"].dropna().unique())[:15])

        if merged:
            sample = (
                df[["Competition_norm", "Multiplier"]]
                .drop_duplicates()
                .sort_values("Competition_norm")
                .head(15)
            )
            print("[DEBUG] Sample multiplier matches:")
            print(sample.to_string(index=False))

            unmatched = df.loc[df["Multiplier"].eq(1.0) & df["Competition_norm"].notna(), "Competition_norm"].unique()
            if len(unmatched) > 0:
                print("[DEBUG] ⚠️ Competitions missing multipliers (first 10):", unmatched[:10])

    except Exception as e:
        print(f"[DEBUG] ⚠️ Failed to merge league multipliers: {e}")
        df["Multiplier"] = 1.0

    # ============================================================
    # 🪪 3. Rename Identifiers to Match Radar Columns
    # ============================================================
    rename_map = {}
    if "Name" in df.columns:
        rename_map["Name"] = "Player"
    if "Primary Position" in df.columns:
        rename_map["Primary Position"] = "Position"
    if "Minutes" in df.columns:
        rename_map["Minutes"] = "Minutes played"

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

    # ============================================================
    # ⚙️ 4. Derived / Calculated Metrics
    # ============================================================
    cross_cols = [c for c in df.columns if "crosses" in c.lower()]
    crossperc_cols = [c for c in df.columns if "crossing%" in c.lower()]
    if cross_cols and crossperc_cols:
        df["Successful Crosses"] = (
            pd.to_numeric(df[cross_cols[0]], errors="coerce") *
            (pd.to_numeric(df[crossperc_cols[0]], errors="coerce") / 100.0)
        )

    if "Player Season Total Dribbles 90" in df.columns and "Player Season Failed Dribbles 90" in df.columns:
        df["Successful Dribbles"] = (
            pd.to_numeric(df["Player Season Total Dribbles 90"], errors="coerce").fillna(0)
            - pd.to_numeric(df["Player Season Failed Dribbles 90"], errors="coerce").fillna(0)
        )

    # ============================================================
    # 🧩 5. Position Handling & Mapping
    # ============================================================
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

    # Map to six positional groups
    if "Position" in df.columns:
        df["Six-Group Position"] = df["Position"].apply(map_first_position_to_group)
    else:
        df["Six-Group Position"] = np.nan

    # Duplicate generic CMs into both 6 & 8
    if "Six-Group Position" in df.columns:
        cm_mask = df["Six-Group Position"] == "Centre Midfield"
        if cm_mask.any():
            cm_rows = df.loc[cm_mask].copy()
            cm_as_6 = cm_rows.copy()
            cm_as_6["Six-Group Position"] = "Number 6"
            cm_as_8 = cm_rows.copy()
            cm_as_8["Six-Group Position"] = "Number 8"
            df = pd.concat([df, cm_as_6, cm_as_8], ignore_index=True)

    # ============================================================
    # ✅ 6. Return Cleaned Data
    # ============================================================
    return df

# ---------- Cached Data Loader ----------
@st.cache_data(show_spinner=True)
def load_data_once():
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
df_all_raw = load_data_once()

if df_all_raw is None or df_all_raw.empty:
    st.error("❌ No player data loaded. Check your StatsBomb CSV path or contents.")
    st.stop()

if "Competition" not in df_all_raw.columns:
    st.error("❌ Expected a 'Competition' column in your data.")
    st.stop()

print("[DEBUG] Sample Competitions:", df_all_raw["Competition"].dropna().unique()[:10])

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
league_candidates = ["Competition_norm", "Competition", "competition_norm", "competition"]
league_col = next((c for c in league_candidates if c in df.columns), None)
if league_col is None:
    st.error("❌ No league/competition column found after preprocessing.")
    st.stop()

# build clean options; avoid the literal string "nan"
leagues_series = pd.Series(df[league_col], dtype="string").str.strip()
all_leagues = sorted([x for x in leagues_series.dropna().unique().tolist() if x and x.lower() != "nan"])

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

valid_defaults = [l for l in st.session_state.get("league_selection", []) if l in all_leagues]

selected_leagues = st.multiselect(
    "Leagues",
    options=all_leagues,
    default=valid_defaults,
    key="league_selection",
    label_visibility="collapsed"
)

# keep session_state clean if options changed
if set(valid_defaults) != set(st.session_state.league_selection):
    st.session_state.league_selection = valid_defaults

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
        st.session_state.min_minutes = 500

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
    import matplotlib.colors as mcolors

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

def generate_player_summary(player_name: str, plot_data: pd.DataFrame, metrics: dict):
    """Generate a realistic, scout-style summary in Tom's critical tone using OpenAI (GPT-4o-mini)."""
    try:
        row = plot_data.loc[plot_data["Player"] == player_name].iloc[0]
    except IndexError:
        return "No data available for this player."

    role = str(row.get("Six-Group Position", "player"))
    league = str(row.get("Competition_norm", ""))
    age = row.get("Age", "")
    team = str(row.get("Team", ""))
    mins = row.get("Minutes played", 0)

    # Collect numeric metrics
    metric_percentiles = {
        m: row.get(f"{m} (percentile)", np.nan)
        for m in metrics.keys()
        if f"{m} (percentile)" in row.index
    }
    metric_text = ", ".join([f"{k}: {v:.0f}" for k, v in metric_percentiles.items() if pd.notnull(v)])

    # --- Build dynamic prompt ---
    prompt = f"""
    You are writing a detailed but concise player scouting report in the style of Tom Irving,
    a professional football recruitment analyst known for honest, critical, and realistic assessments.

    Write a 5–6 sentence paragraph about {player_name}, a {role.lower()} aged {age}, currently playing in {league} for {team}.

    You have access to percentile data (0–100) for performance metrics:
    {metric_text}

    Tone and writing style rules:
    - Do NOT start with cliches like “is an exciting” or “is a talented” player.
    - Vary the opening line. It can start with what kind of player he looks like, what stands out, or even what’s missing.
    - If metrics are low (below 40th percentile), acknowledge weaknesses clearly. Use natural phrasing like:
        • “Struggles to impact games consistently.” 
        • “Can look limited when the game becomes physical.”
        • “Output doesn’t yet match his effort.”
    - If metrics are high (above 70th percentile), highlight them naturally:
        • “Ranks among the best in his league for dribbles and chance creation.”
        • “Shows real control under pressure and moves play forward quickly.”
    - Keep a balanced tone — be fair, but never overly generous. You're not writing marketing material.
    - Combine data insight with realistic football language (movement, body shape, pressing work, mentality).
    - Vary phrasing so no two reports feel copy-pasted.
    - End with one strong, definitive sentence that sums up his player type or potential fit — e.g.:
        • “A physically strong, low-risk defender who could suit a compact system.”
        • “A creative wide player with flashes of quality but inconsistent end product.”
        • “Profiles as a hard-working forward who fits the pressing style but lacks a ruthless edge.”

    Write in Tom’s natural tone, as seen in these examples:
    - “He’s got a good base technically but sometimes forces play when it’s not on.”
    - “Not the most dynamic athlete, but his awareness and timing stand out.”
    - “There’s something raw but promising about him — a player who could develop quickly in the right setup.”

    Be honest, concise, and analytical. Avoid repetition. Write like a human scout.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            temperature=0.85,  # slightly higher for more creative, human tone
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ AI summary generation failed: {e}"

# ---------- Plot ----------
if st.session_state.selected_player:
    plot_radial_bar_grouped(
        st.session_state.selected_player,
        plot_data,
        metric_groups,
        group_colors
    )

# ---------- AI Scouting Summary ----------
st.markdown("### 🧠 AI Scouting Summary")

if st.button("Generate AI Summary", key="ai_summary_button"):
    with st.spinner("Generating AI scouting report..."):
        summary_text = generate_player_summary(
            st.session_state.selected_player,
            plot_data,
            metric_groups
        )
        st.markdown(summary_text)

# ---------- 🔍 Find 10 Similar Players ----------
st.markdown("### 🔍 Find Similar Players")

MINUTES_SIMILAR = 400
pos_col = "Six-Group Position"
minutes_col = "Minutes played"

def find_similar_players_same_scale(
    player_name: str,
    base_df: pd.DataFrame,
    metrics: list,
    position_col: str,
    lower_is_better: set,
    n_similar: int = 10,
    minutes_floor: int = 400
):
    """Find most similar players using the *existing* Weighted Z Score scale and same 0–100 logic."""
    if player_name not in base_df["Player"].values:
        return pd.DataFrame(), f"{player_name} not found."

    # Selected player's position
    pos = base_df.loc[base_df["Player"] == player_name, position_col].iloc[0]
    if pd.isna(pos) or not pos:
        return pd.DataFrame(), f"No position found for {player_name}."

    # Filter pool: same position & ≥ minutes_floor
    mins_num = pd.to_numeric(base_df.get(minutes_col, np.nan), errors="coerce")
    pool = base_df[
        (base_df[position_col] == pos) &
        (mins_num >= minutes_floor)
    ].copy()
    if pool.empty:
        return pd.DataFrame(), f"No players (≥{minutes_floor} mins) at {pos}."

    # Ensure numeric metrics exist
    valid_metrics = [m for m in metrics if m in pool.columns]
    if not valid_metrics:
        return pd.DataFrame(), "No valid metrics for similarity."

    # Z-score within the pool (so distances are scale-free)
    X = pool[valid_metrics].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    Z = (X - X.mean()) / X.std(ddof=0).replace(0, np.nan)
    Z = Z.fillna(0)

    # Get player vector
    pZ = Z.loc[pool["Player"] == player_name]
    if pZ.empty:
        return pd.DataFrame(), f"{player_name} has no valid metric data."

    # Compute Euclidean distance
    diff = Z.values - pZ.values[0]
    dists = np.sqrt(np.sum(diff**2, axis=1))
    pool["Similarity Score"] = 100.0 - (dists / dists.max() * 100.0 if dists.max() != 0 else 0)

    # Exclude self
    out = (
        pool[pool["Player"] != player_name]
        .sort_values("Similarity Score", ascending=False)
        .head(n_similar)
        .copy()
    )

    # Keep same “Score (0–100)” as ranking table (don’t recalc)
    score_map = base_df.set_index("Player")["Score (0–100)"].to_dict()
    out["Score (0–100)"] = out["Player"].map(score_map)

    # Clean columns
    out.rename(columns={"Team within selected timeframe": "Team",
                        "Competition_norm": "League"}, inplace=True)
    keep_cols = ["Player", "Team", "League", "Age", minutes_col, "Score (0–100)", "Similarity Score"]
    for c in keep_cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[keep_cols]

    # Round + tidy
    out[minutes_col] = pd.to_numeric(out[minutes_col], errors="coerce").fillna(0).astype(int)
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce").round(0)
    out["Score (0–100)"] = pd.to_numeric(out["Score (0–100)"], errors="coerce").round(1)
    out["Similarity Score"] = pd.to_numeric(out["Similarity Score"], errors="coerce").round(1)

    return out, None


# ---------- Run Button ----------
if st.button("Find 10 Similar Players", key="similar_players_button"):
    with st.spinner("Finding most similar players..."):
        similar_df, err = find_similar_players_same_scale(
            st.session_state.selected_player,
            plot_data,  # ✅ use same scaled dataset that feeds the ranking table
            position_metrics[st.session_state.template_select]["metrics"],
            pos_col,
            {"Turnovers", "Fouls", "Pr. Long Balls", "UPr. Long Balls"},
            n_similar=10,
            minutes_floor=MINUTES_SIMILAR,
        )

    if err:
        st.warning(err)
    elif similar_df.empty:
        st.info("No similar players found.")
    else:
        st.markdown(f"#### 10 Players Most Similar to {st.session_state.selected_player}")

        # Remove duplicate cols safely
        similar_df = similar_df.loc[:, ~similar_df.columns.duplicated()]

        # Clean order
        similar_df = similar_df[["Player", "Team", "League", "Age", "Minutes played", "Score (0–100)", "Similarity Score"]]

        st.dataframe(similar_df, use_container_width=True)
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
# 🟢 LOAD FAVOURITES FROM SUPABASE AND APPLY COLOURS
# ============================================================

# --- Load current favourites ---
from lib.favourites_repo import get_supabase_client

def get_favourites_with_colours_live():
    """Fetch favourites (shared cloud data) — no caching, no globals."""
    sb = get_supabase_client()
    if sb is None:
        return {}

    try:
        res = sb.table("favourites").select("*").execute()
        if not res.data:
            return {}
        return {
            r.get("player"): {
                "colour": r.get("colour", ""),
                "comment": r.get("comment", ""),
                "visible": bool(r.get("visible", True)),
            }
            for r in res.data
            if r.get("player")
        }
    except Exception as e:
        st.warning(f"⚠️ Could not load favourites: {e}")
        return {}

# --- Build local dictionary for colouring ---
favs = get_favourites_with_colours_live()

print("[DEBUG] Sample favs from Supabase:")
print({k: v for k, v in list(favs.items())[:5]})

# --- Colour emoji map ---
COLOUR_EMOJI = {
    "🟣 Needs Checked": "🟣",
    "🟡 Monitor": "🟡",
    "🟢 Go": "🟢",
    "🟠 Out Of Reach": "🟠",
    "🔴 No Further Interest": "🔴",
    "Needs Checked": "🟣",
    "Monitor": "🟡",
    "Go": "🟢",
    "No Further Interest": "🔴",
    "🟣": "🟣",
    "🟡": "🟡",
    "🟢": "🟢",
    "🟠": "🟠",
    "🔴": "🔴",
}

def colourize_player_name(name: str, favs_dict: dict) -> str:
    """Attach correct emoji to player name."""
    data = favs_dict.get(name)
    if not data:
        return name
    emoji = COLOUR_EMOJI.get(str(data.get("colour", "")).strip(), "")
    return f"{emoji} {name}" if emoji else name

# ============================================================
# 🧾 ENSURE TABLE COLUMNS EXIST & REORDER
# ============================================================
z_ranking["Player (coloured)"] = z_ranking["Player"].apply(lambda n: colourize_player_name(n, favs))
z_ranking["⭐ Favourite"] = z_ranking["Player"].apply(lambda n: bool(favs.get(n, {}).get("visible", False)))

required_cols = [
    "⭐ Favourite", "Player (coloured)", "Positions played", "Team", "League",
    "Multiplier", "Score (0–100)", "Age", "Minutes played", "Rank"
]
for col in required_cols:
    if col not in z_ranking.columns:
        z_ranking[col] = np.nan
z_ranking = z_ranking[required_cols]

# ============================================================
# 📋 EDITABLE TABLE (NO CACHING SO FILTERS ALWAYS REFRESH)
# ============================================================

# Build a stable signature so Streamlit resets editor state when filters change
sig_parts = (
    tuple(sorted(selected_leagues)),
    int(min_minutes),
    tuple(selected_groups),
    selected_position_template,
    len(z_ranking),                         # size change
    float(z_ranking["Score (0–100)"].sum()) # quick content checksum
)
editor_key = f"ranking_editor_{hash(sig_parts)}"

edited_df = st.data_editor(
    z_ranking,   # <- always the fresh, filtered table
    column_config={
        "Player (coloured)": st.column_config.TextColumn(
            "Player", help="Shows Favourite colour (🟢🟡🔴🟣 only if marked)"
        ),
        "⭐ Favourite": st.column_config.CheckboxColumn(
            "⭐ Favourite", help="Mark or unmark as favourite (shared to Supabase)"
        ),
        "Multiplier": st.column_config.NumberColumn(
            "League Weight", help="League weighting applied in ranking", format="%.3f"
        ),
    },
    hide_index=False,
    width="stretch",
    key=editor_key,
)

# ============================================================
# 💾 APPLY CHANGES TO SUPABASE + SMART GOOGLE SHEET LOGGING
# ============================================================
import time

# Cache favourites for smoother reloads (prevents needless reruns)
@st.cache_data(ttl=5, show_spinner=False)
def load_favourites_cached():
    return get_favourites_with_colours_live()

favs_live = load_favourites_cached()

print("[DEBUG] === Sync section triggered ===")
print(f"[DEBUG] Total rows in table: {len(edited_df)}")

# --- Run guard to stop immediate re-execution ---
if st.session_state.get("_last_sync_time") and time.time() - st.session_state["_last_sync_time"] < 3:
    print("[DEBUG] Skipping sync — triggered too soon after last run")
else:
    st.session_state["_last_sync_time"] = time.time()

    if "⭐ Favourite" not in edited_df.columns:
        st.warning("⚠️ Could not find the '⭐ Favourite' column — skipping sync.")
    else:
        favourite_rows = edited_df[edited_df["⭐ Favourite"] == True].copy()
        print(f"[DEBUG] Favourites to sync: {len(favourite_rows)} of {len(edited_df)}")

        # ---- UPSERT NEW OR UPDATED FAVOURITES ----
        deleted_players = {p for p, d in favs_live.items() if not d.get("visible", True)}

        for _, row in favourite_rows.iterrows():
            player_raw = str(row.get("Player (coloured)", "")).strip()
            player_name = re.sub(r"^[🟢🟡🔴🟣]\s*", "", player_raw).strip()
            team = row.get("Team", "")
            league = row.get("League", "")
            position = row.get("Positions played", "")

            prev_data = favs_live.get(player_name, {})
            prev_visible = bool(prev_data.get("visible", False))

            # 🧠 Skip if player was deleted (hidden) and not manually re-starred
            if player_name in deleted_players and not prev_visible:
                print(f"[DEBUG] Skipping {player_name} — hidden in Supabase (won’t auto-revive)")
                continue

            # 🧠 Only upsert if not already visible in Supabase
            if not prev_visible:
                payload = {
                    "player": player_name,
                    "team": team,
                    "league": league,
                    "position": position,
                    "colour": prev_data.get("colour", "🟣 Needs Checked"),
                    "comment": prev_data.get("comment", ""),
                    "visible": True,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "source": "radar-page",
                }

                upsert_favourite(payload, log_to_sheet=True)
                print(f"[LOG] ✅ Added or reactivated {player_name}")
            else:
                print(f"[DEBUG] Skipping {player_name} — already visible in Supabase")

        # ---- HIDE UNSTARRED FAVOURITES ----
        non_fav_rows = edited_df[edited_df["⭐ Favourite"] == False]
        for _, row in non_fav_rows.iterrows():
            player_raw = str(row.get("Player (coloured)", "")).strip()
            player_name = re.sub(r"^[🟢🟡🔴🟣]\s*", "", player_raw).strip()

            old_visible = favs_live.get(player_name, {}).get("visible", False)
            if old_visible:
                hide_favourite(player_name)
                print(f"[INFO] Hid favourite: {player_name}")
