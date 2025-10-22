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

    # --- Slovakia ---
    "1. Liga": "Slovakia 1. Liga",

    # --- Slovenia ---
    "1. Liga": "Slovenia 1. Liga",

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
    """Add numeric Age column based on birth_date or Birth Date (if present)."""
    today = datetime.today()

    # --- Detect correct birth date column ---
    birth_col = None
    for c in df.columns:
        if c.strip().lower() in {"birth_date", "birth date"}:
            birth_col = c
            break

    if not birth_col:
        df["Age"] = np.nan
        print("[DEBUG] No birth date column found — Age set to NaN")
        return df

    df["Age"] = pd.to_datetime(df[birth_col], errors="coerce").apply(
        lambda dob: today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        if pd.notna(dob) else np.nan
    )

    print(f"[DEBUG] Age column created from '{birth_col}'. Non-null ages: {df['Age'].notna().sum()}")
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
    # ⚖️ 2. Merge League Multipliers (by Competition_ID first, fallback to name)
    # ============================================================
    try:
        multipliers_path = ROOT_DIR / "league_multipliers.xlsx"
        m = pd.read_excel(multipliers_path)

        # --- Clean multipliers file ---
        m.columns = m.columns.str.strip().str.lower().str.replace(" ", "_")
        m.rename(columns={
            "competitionid": "competition_id",
            "competition_id_": "competition_id",
            "competition": "league",
            "league_name": "league"
        }, inplace=True)

        m["competition_id"] = pd.to_numeric(m.get("competition_id", np.nan), errors="coerce").astype("Int64")
        m["multiplier"] = pd.to_numeric(m.get("multiplier", 1.0), errors="coerce").fillna(1.0)
        m["league"] = m.get("league", "").astype(str).str.strip()

        # --- Normalise competition ID in main data ---
        id_candidates = [
            "Competition_ID", "competition_id", "Competition Id", "Competition id"
        ]
        found_id = next((c for c in id_candidates if c in df.columns), None)
        if found_id:
            df.rename(columns={found_id: "competition_id"}, inplace=True)
            df["competition_id"] = pd.to_numeric(df["competition_id"], errors="coerce").astype("Int64")
        else:
            df["competition_id"] = pd.NA

        # --- Merge by ID first ---
        df = df.merge(m[["competition_id", "multiplier"]], on="competition_id", how="left")

        # --- Fallback: merge by league name if no matches found ---
        if df["multiplier"].isna().all():
            df = df.merge(m[["league", "multiplier"]], left_on="Competition_norm", right_on="league", how="left")

        # --- Final numeric Multiplier column ---
        df["Multiplier"] = pd.to_numeric(df["multiplier"], errors="coerce").fillna(1.0)
        df.drop(columns=["multiplier", "league"], inplace=True, errors="ignore")

    except Exception:
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

    if "Team within selected timeframe" not in df.columns:
        df["Team within selected timeframe"] = df["Team"] if "Team" in df.columns else np.nan
    if "Height" not in df.columns:
        df["Height"] = np.nan

    # Map base position
    if "Position" in df.columns:
        df["Six-Group Position"] = df["Position"].apply(map_first_position_to_group)
    else:
        df["Six-Group Position"] = np.nan

    # ✅ Only duplicate true Centre Midfielders once (for each unique player-team combo)
    if "Six-Group Position" in df.columns:
        cm_mask = df["Six-Group Position"].eq("Centre Midfield")

        if cm_mask.any():
            # Deduplicate by player + team (avoids multiple club rows repeating)
            cm_rows = (
                df.loc[cm_mask, ["Player", "Team", "Six-Group Position"]]
                .drop_duplicates(subset=["Player", "Team"])
            )
            if not cm_rows.empty:
                cm_as_6 = df.loc[
                    df["Player"].isin(cm_rows["Player"])
                    & df["Team"].isin(cm_rows["Team"])
                    & cm_mask
                ].copy()
                cm_as_8 = cm_as_6.copy()

                cm_as_6["Six-Group Position"] = "Number 6"
                cm_as_8["Six-Group Position"] = "Number 8"

                # Only add if they don't already exist
                already_6_8 = df[
                    (df["Six-Group Position"].isin(["Number 6", "Number 8"]))
                    & df["Player"].isin(cm_rows["Player"])
                ]
                new_rows = pd.concat([cm_as_6, cm_as_8], ignore_index=True)
                new_rows = new_rows[
                    ~new_rows.set_index(["Player", "Team", "Six-Group Position"]).index.isin(
                        already_6_8.set_index(["Player", "Team", "Six-Group Position"]).index
                    )
                ]

                df = pd.concat([df, new_rows], ignore_index=True)

    return df
# ---------- Cached Data Loader ----------
@st.cache_data(show_spinner=True)
def load_data_once():
    """Load and preprocess StatsBomb data once per session."""
    path = DATA_PATH
    sig = _data_signature(path)

    # ============================================================
    # 1️⃣ LOAD FILE(S)
    # ============================================================
    if path.is_file():
        df_raw = load_one_file(path)
    else:
        files = sorted(
            f for f in path.iterdir()
            if f.is_file() and (f.suffix.lower() in {".csv", ".xlsx", ".xls"} or f.suffix == "")
        )
        if not files:
            st.error(f"No data files found in {path.name}. Please add a CSV or XLSX file.")
            st.stop()

        frames = []
        for f in files:
            try:
                frames.append(load_one_file(f))
            except Exception:
                continue

        if not frames:
            st.error("No readable player data files found.")
            st.stop()

        df_raw = pd.concat(frames, ignore_index=True, sort=False)

    # ============================================================
    # 2️⃣ AGE + PREPROCESSING
    # ============================================================
    df_raw = add_age_column(df_raw)
    df_preprocessed = preprocess_df(df_raw)
    return df_preprocessed


# ============================================================
# ✅ Load + prepare the data
# ============================================================
df_all_raw = load_data_once()

if df_all_raw is None or df_all_raw.empty:
    st.error("❌ No player data loaded. Check your StatsBomb CSV path or contents.")
    st.stop()

if "Competition" not in df_all_raw.columns:
    st.error("❌ Expected a 'Competition' column in your data.")
    st.stop()

# ---------- Clean and prepare ----------
df_all_raw.columns = (
    df_all_raw.columns.astype(str)
    .str.strip()
    .str.replace(u"\xa0", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
)

# ✅ No extra “Birth Date” handling block needed anymore

# ---------- Preprocess and create the working copy ----------
df_all = df_all_raw.copy()
df = df_all.copy()

# ============================================================
# 3️⃣ LEAGUE FILTER
# ============================================================
league_candidates = ["Competition_norm", "Competition", "competition_norm", "competition"]
league_col = next((c for c in league_candidates if c in df.columns), None)
if league_col is None:
    st.error("❌ No league/competition column found after preprocessing.")
    st.stop()

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

if set(valid_defaults) != set(st.session_state.league_selection):
    st.session_state.league_selection = valid_defaults

if selected_leagues:
    df = df[df[league_col].isin(selected_leagues)].copy()
    st.caption(f"Leagues selected: {len(selected_leagues)} | Players: {len(df)}")
    if df.empty:
        st.warning("No players match the selected leagues. Try a different selection.")
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

# ---------- Build metric pool for Essential Criteria ----------
current_template_name = st.session_state.template_select or list(position_metrics.keys())[0]
current_metrics = position_metrics[current_template_name]["metrics"]

for m in current_metrics:
    if m not in df.columns:
        df[m] = 0
df[current_metrics] = df[current_metrics].fillna(0)

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
pos_col = "Six-Group Position"

# Compute Z per metric: (raw - mean)/std per position; invert lower-better
raw_z_all = pd.DataFrame(index=df_all.index, columns=sel_metrics, dtype=float)
for m in sel_metrics:
    df_all[m] = pd.to_numeric(df_all[m], errors="coerce").fillna(0)
    z_per_group = df_all.groupby(pos_col)[m].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
    )
    if m in LOWER_IS_BETTER:
        z_per_group *= -1
    raw_z_all[m] = z_per_group.fillna(0)

# ---------- Average + Weighted Z (sign-aware) ----------
avg_z_all = raw_z_all.mean(axis=1).fillna(0)
df_all["Avg Z Score"] = avg_z_all

mult = pd.to_numeric(df_all.get("Multiplier", 1.0), errors="coerce").fillna(1.0)
avg_z = df_all["Avg Z Score"]

# ✅ Apply proper weighting logic (strong leagues boost, weak leagues dampen)
df_all["Weighted Z Score"] = np.select(
    [avg_z > 0, avg_z < 0],
    [avg_z * mult, avg_z / mult],
    default=0.0
)

# ---------- LFC Weighted Z (Scottish Premiership 1.20) ----------
df_all["LFC Multiplier"] = mult
df_all.loc[df_all["Competition_norm"] == "Scotland Premiership", "LFC Multiplier"] = 1.20
lfc_mult = df_all["LFC Multiplier"]

df_all["LFC Weighted Z"] = np.select(
    [avg_z > 0, avg_z < 0],
    [avg_z * lfc_mult, avg_z / lfc_mult],
    default=0.0
)

# ---------- Anchors + Scaling ----------
_mins_all = pd.to_numeric(df_all.get("Minutes played", np.nan), errors="coerce")
eligible = df_all[_mins_all >= 600].copy()
if eligible.empty:
    eligible = df_all.copy()

anchors = (
    eligible.groupby(pos_col, dropna=False)["Weighted Z Score"]
    .agg(_scale_min="min", _scale_max="max")
    .fillna(0)
)
df_all = df_all.merge(anchors, left_on=pos_col, right_index=True, how="left")

def _to100(v, lo, hi):
    if pd.isna(v) or pd.isna(lo) or pd.isna(hi) or hi <= lo:
        return 50.0
    return np.clip((v - lo) / (hi - lo) * 100.0, 0.0, 100.0)

df_all["Score (0–100)"] = [
    _to100(v, lo, hi)
    for v, lo, hi in zip(df_all["Weighted Z Score"], df_all["_scale_min"], df_all["_scale_max"])
]
df_all["LFC Score (0–100)"] = [
    _to100(v, lo, hi)
    for v, lo, hi in zip(df_all["LFC Weighted Z"], df_all["_scale_min"], df_all["_scale_max"])
]

df_all[["Score (0–100)", "LFC Score (0–100)"]] = (
    df_all[["Score (0–100)", "LFC Score (0–100)"]]
    .apply(pd.to_numeric, errors="coerce")
    .round(1)
    .fillna(0)
)

# ---------- Rank ----------
plot_data.sort_values("Weighted Z Score", ascending=False, inplace=True, ignore_index=True)
plot_data["Rank"] = np.arange(1, len(plot_data) + 1)

# ---------- Table ----------
cols_for_table = [
    "Player", "Six-Group Position", "Positions played",
    "Team", league_col, "Multiplier",
    "Avg Z Score", "Weighted Z Score", "LFC Weighted Z",
    "Score (0–100)", "LFC Score (0–100)",
    "Age", "Minutes played", "Rank in Team"
]
for c in cols_for_table:
    if c not in df_team.columns:
        df_team[c] = np.nan

z_ranking = df_team[cols_for_table].copy()
z_ranking.rename(columns={
    "Six-Group Position": "Position",
    league_col: "League",
    "Multiplier": "League Weight",
    "Avg Z Score": "Z Avg",
    "Weighted Z Score": "Z Weighted",
    "LFC Weighted Z": "Z LFC Weighted"
}, inplace=True)

    z_ranking["Age"] = pd.to_numeric(z_ranking["Age"], errors="coerce").round(0)
    z_ranking["Minutes played"] = pd.to_numeric(z_ranking["Minutes played"], errors="coerce").fillna(0).astype(int)
    z_ranking["League Weight"] = pd.to_numeric(z_ranking["League Weight"], errors="coerce").fillna(1.0).round(3)

    favs_in_db = {row[0] for row in get_favourites()}
    z_ranking["⭐ Favourite"] = z_ranking["Player"].isin(favs_in_db)

    edited = st.data_editor(
        z_ranking,
        column_config={
            "⭐ Favourite": st.column_config.CheckboxColumn("⭐ Favourite", help="Mark as favourite"),
            "League Weight": st.column_config.NumberColumn("League Weight", help="League weighting applied in ranking", format="%.3f"),
            "Z Avg": st.column_config.NumberColumn("Z Avg", format="%.3f"),
            "Z Weighted": st.column_config.NumberColumn("Z Weighted", format="%.3f"),
            "Z LFC Weighted": st.column_config.NumberColumn("Z LFC Weighted", format="%.3f"),
            "Score (0–100)": st.column_config.NumberColumn("Score (0–100)", format="%.1f"),
            "LFC Score (0–100)": st.column_config.NumberColumn("LFC Score (0–100)", format="%.1f"),
        },
        hide_index=False,
        width="stretch",
    )

    for _, r in edited.iterrows():
        p = r["Player"]
        if r["⭐ Favourite"] and p not in favs_in_db:
            add_favourite(p, r.get("Team"), r.get("League"), r.get("Positions played"))
        elif not r["⭐ Favourite"] and p in favs_in_db:
            remove_favourite(p)

except Exception as e:
    st.error(f"❌ Could not load data: {e}")

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
    "Multiplier", "Avg Z Score", "Weighted Z Score",
    "Score (0–100)", "LFC Score (0–100)",
    "Age", "Minutes played", "Rank"
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
    z_ranking,
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
        "Avg Z Score": st.column_config.NumberColumn("Avg Z", format="%.3f"),
        "Weighted Z Score": st.column_config.NumberColumn("Weighted Z", format="%.3f"),
        "LFC Score (0–100)": st.column_config.NumberColumn("LFC Score (0–100)", format="%.1f"),
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
            player_name = re.sub(r"^[🟢🟡🔴🟣🟠]\s*", "", player_raw).strip()
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
