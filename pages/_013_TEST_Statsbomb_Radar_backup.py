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

st.set_page_config(page_title="Livingston FC Recruitment App", layout="centered")

# ---------- Authentication ----------
if not check_password():
    st.stop()

# ---------- Branding ----------
show_branding()
st.title("Statsbomb Radar (Test / Backup)")

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
    # Attacking mids → 8
    "CENTREATTACKINGMIDFIELDER": "Number 8", "ATTACKINGMIDFIELDER": "Number 8", "RIGHTATTACKINGMIDFIELDER": "Number 8", "LEFTATTACKINGMIDFIELDER": "Number 8",
    # Wingers / wide mids
    "RIGHTWING": "Winger", "LEFTWING": "Winger",
    "RIGHTMIDFIELDER": "Winger", "LEFTMIDFIELDER": "Winger",
    # Strikers
    "CENTREFORWARD": "Striker", "RIGHTCENTREFORWARD": "Striker", "LEFTCENTREFORWARD": "Striker", "SECONDSTRIKER": "Striker", "10": "Striker",
}

def parse_first_position(cell) -> str:
    if pd.isna(cell):
        return ""
    return _clean_pos_token(str(cell))

def map_first_position_to_group(primary_pos_cell) -> str:
    tok = parse_first_position(primary_pos_cell)
    return RAW_TO_SIX.get(tok, None)  # do not force into Winger

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
DATA_PATH = ROOT_DIR / "statsbomb_player_stats_all_seasons.csv"

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

                # Only add if they do not already exist
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
# ✅ Load + prepare the data (TOP-LEVEL DATA PREP)
# ============================================================
df_all_raw = load_data_once()

# --- Column normaliser helper ---
def _rename_first(df, candidates, target):
    for c in candidates:
        if c in df.columns:
            if c != target:
                df.rename(columns={c: target}, inplace=True)
                print(f"[DEBUG] Renamed '{c}' -> '{target}'")
            return True
    return False

cols_lower = {c.lower(): c for c in df_all_raw.columns}  # map to actual case

def real(col):
    return col if col in df_all_raw.columns else cols_lower.get(col, col)

# Season
_rename_first(
    df_all_raw,
    [real("Season"), real("season_name"), real("season")],
    "Season",
)

# Competition
_rename_first(
    df_all_raw,
    [real("Competition"), real("competition_name"), real("competition")],
    "Competition",
)

# Team
_rename_first(
    df_all_raw,
    [real("Team"), real("team_name")],
    "Team",
)

# Player
_rename_first(
    df_all_raw,
    [real("Player"), real("player_name"), real("Name")],
    "Player",
)

# Positions
_rename_first(
    df_all_raw,
    [real("Primary Position"), real("primary_position")],
    "Position",
)
_rename_first(
    df_all_raw,
    [real("Secondary Position"), real("secondary_position")],
    "Secondary Position",
)

# Minutes
_rename_first(
    df_all_raw,
    [real("Minutes played"), real("Minutes"), real("player_season_minutes")],
    "Minutes played",
)

# Competition_ID (helps league multipliers)
_rename_first(
    df_all_raw,
    [real("Competition_ID"), real("competition_id"), real("Competition Id"), real("Competition id")],
    "competition_id",
)

# Clean column names
df_all_raw.columns = (
    df_all_raw.columns.astype(str)
    .str.replace("\xa0", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# --- Normalise season column names / values ---

# These are the column names that exist in your file
season_priority = ["season_name", "season", "season_id"]

# Find the first one that exists
found_season_col = next((c for c in season_priority if c in df_all_raw.columns), None)

if not found_season_col:
    st.error("❌ No season column was found. Expected one of: season_name, season, season_id")
    st.stop()

# Rename it to Season if needed
if found_season_col != "Season":
    df_all_raw.rename(columns={found_season_col: "Season"}, inplace=True)
    print(f"[DEBUG] Renamed '{found_season_col}' → 'Season'")

# Final safety check
if "Season" not in df_all_raw.columns:
    st.error("❌ Season column still missing after rename. Cannot continue.")
    st.stop()

# Convert to clean strings
df_all_raw["Season"] = df_all_raw["Season"].astype(str).str.strip()

# ---------- Create Season_norm ----------
def normalise_season_label(s: str) -> str:
    s = s.strip()
    if "/" in s:
        return s
    if re.match(r"^\d{4}$", s):
        return f"{s}/{int(s) + 1}"
    return s

if "Season" in df_all_raw.columns:
    df_all_raw["Season_norm"] = df_all_raw["Season"].astype(str).apply(normalise_season_label)
else:
    df_all_raw["Season_norm"] = np.nan

# ---------- Build PlayerKey (for duplicate names across seasons) ----------
def add_player_key(df: pd.DataFrame) -> pd.DataFrame:
    """Create a PlayerKey using player_id if present, else Player + DOB, else Player."""
    # Try ID
    id_col = None
    for c in df.columns:
        if c.lower() in {"player_id", "playerid", "id_player"}:
            id_col = c
            break

    birth_col = None
    for c in df.columns:
        cl = c.lower()
        if "birth" in cl and "date" in cl:
            birth_col = c
            break

    if id_col:
        df["PlayerKey"] = df[id_col].astype(str).str.strip()
    elif "Player" in df.columns and birth_col:
        dob_series = pd.to_datetime(df[birth_col], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
        df["PlayerKey"] = df["Player"].astype(str).str.strip() + "|" + dob_series
    elif "Player" in df.columns:
        df["PlayerKey"] = df["Player"].astype(str).str.strip()
    else:
        df["PlayerKey"] = df.index.astype(str)

    return df

df_all_raw = add_player_key(df_all_raw)

# ---------- Basic inspection ----------
st.write("🔍 Columns in df_all_raw:", list(df_all_raw.columns))

if df_all_raw is None or df_all_raw.empty:
    st.error("❌ No player data loaded. Check your StatsBomb CSV path or contents.")
    st.stop()

if "Competition" not in df_all_raw.columns:
    st.error("❌ Expected a 'Competition' column in your data.")
    st.stop()

# ---------- Preprocess and create working copy ----------
df_all = df_all_raw.copy()
df_all = preprocess_df(df_all)  # already handles Competition_norm, positions, multipliers etc
df = df_all.copy()

# ============================================================
# LEAGUE FILTER
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
    st.caption(f"Leagues selected: {len(selected_leagues)} | Players rows: {len(df)}")
    if df.empty:
        st.warning("No players match the selected leagues. Try a different selection.")
        st.stop()
else:
    st.info("No leagues selected. Pick at least one or click ‘Select all’.")
    st.stop()

# ============================================================
# SEASON FILTER (+ “All (3-Season Avg)” OPTION)
# ============================================================
if "Season_norm" in df_all.columns:
    season_options = sorted(df_all["Season_norm"].dropna().unique().tolist(), reverse=True)
    # Insert synthetic “All (3-Season Avg)” option
    season_options_with_all = ["All (3-Season Avg)"] + season_options

    selected_season = st.selectbox("Select Season or 3-Season View", season_options_with_all)
    st.session_state["selected_season"] = selected_season

    if selected_season == "All (3-Season Avg)":
        st.caption("Showing **3-season averaged** view (where available) and trend information. Radar still uses per-season data.")
        # For the main df used to build radar and filters, keep all seasons for now
        df = df.copy()
    else:
        st.caption(f"Showing data for **{selected_season}** season.")
        df = df_all[
            (df_all["Season_norm"] == selected_season)
            | (df_all["Season"] == selected_season)
        ].copy()
        if df.empty:
            st.warning("No players found for this season selection.")
            st.stop()
else:
    st.info("No 'Season' column found — skipping season filter.")

# ---------- Minutes + Age filters ----------
minutes_col = "Minutes played"
if minutes_col not in df.columns:
    df[minutes_col] = np.nan

c1, c2 = st.columns(2)

with c1:
    if "min_minutes" not in st.session_state:
        st.session_state.min_minutes = 500

    st.session_state.min_minutes = st.number_input(
        "Minimum minutes to include",
        min_value=0,
        value=st.session_state.min_minutes,
        step=50,
        key="min_minutes_input"
    )

    min_minutes = st.session_state.min_minutes

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

st.caption(f"Filtering on '{minutes_col}' ≥ {min_minutes}. Rows remaining: {len(df)}")

# ---------- Position Group Section ----------
st.markdown("#### 🟡 Select Position Group")

available_groups = []
if "Six-Group Position" in df.columns:
    available_groups = [g for g in SIX_GROUPS if g in df["Six-Group Position"].unique()]

if "selected_groups" not in st.session_state:
    st.session_state.selected_groups = []

selected_groups = st.multiselect(
    "Position Groups",
    options=available_groups,
    default=st.session_state.selected_groups,
    label_visibility="collapsed",
    key="pos_group_multiselect"
)

if selected_groups:
    df = df[df["Six-Group Position"].isin(selected_groups)].copy()
    if df.empty:
        st.warning("No players after group filter. Clear filters or choose different groups.")
        st.stop()

current_single_group = selected_groups[0] if len(selected_groups) == 1 else None

# ---------- Template selection ----------
if "template_select" not in st.session_state:
    st.session_state.template_select = list(position_metrics.keys())[0]
if "last_template_choice" not in st.session_state:
    st.session_state.last_template_choice = st.session_state.template_select
if "manual_override" not in st.session_state:
    st.session_state.manual_override = False
if "last_groups_tuple" not in st.session_state:
    st.session_state.last_groups_tuple = tuple()

# Auto-sync template with selected group when only one
if tuple(selected_groups) != st.session_state.last_groups_tuple:
    if len(selected_groups) == 1:
        pos = selected_groups[0]
        if pos in position_metrics:
            st.session_state.template_select = pos
            st.session_state.manual_override = False
    st.session_state.last_groups_tuple = tuple(selected_groups)

st.markdown("#### 📊 Choose Radar Template")
template_names = list(position_metrics.keys())
if st.session_state.template_select not in template_names:
    st.session_state.template_select = template_names[0]

selected_position_template = st.selectbox(
    "Radar Template",
    template_names,
    index=template_names.index(st.session_state.template_select),
    key="template_select",
    label_visibility="collapsed"
)

if st.session_state.template_select != st.session_state.last_template_choice:
    st.session_state.manual_override = True
    st.session_state.last_template_choice = st.session_state.template_select

# ---------- Build metric pool for the selected template ----------
current_template_name = st.session_state.template_select or list(position_metrics.keys())[0]
metrics = position_metrics[current_template_name]["metrics"]
metric_groups = position_metrics[current_template_name]["groups"]

# Make sure metric columns exist and are numeric
for m in metrics:
    if m not in df_all.columns:
        df_all[m] = 0
    if m not in df.columns:
        df[m] = 0
    df_all[m] = pd.to_numeric(df_all[m], errors="coerce").fillna(0)
    df[m] = pd.to_numeric(df[m], errors="coerce").fillna(0)

# ---------- Player list ----------
if "Player" not in df.columns:
    st.error("Expected a 'Player' column in the data.")
    st.stop()

players = df["Player"].dropna().unique().tolist()
if not players:
    st.warning("No players available after filters.")
    st.stop()

if "selected_player" not in st.session_state:
    st.session_state.selected_player = players[0]

if st.session_state.selected_player not in players:
    st.session_state.selected_player = players[0]

selected_player = st.selectbox(
    "Choose a player",
    players,
    index=players.index(st.session_state.selected_player) if st.session_state.selected_player in players else 0,
    key="player_select"
)
st.session_state.selected_player = selected_player

# ============================================================
# PERCENTILES & SCORE CALCULATIONS
# ============================================================
LOWER_IS_BETTER = {
    "Turnovers",
    "Fouls",
    "Pr. Long Balls",
    "UPr. Long Balls",
}

def pct_rank(series: pd.Series, lower_is_better: bool) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").fillna(0)
    r = series.rank(pct=True, ascending=True)
    if lower_is_better:
        p = 1.0 - r
    else:
        p = r
    return (p * 100.0).round(1)

# --- Percentiles for radar (within selected leagues vs pooled) ---
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
            percentile_df_chart[m] = 50.0
else:
    for m in metrics:
        try:
            percentile_df_chart[m] = pct_rank(df[m], lower_is_better=(m in LOWER_IS_BETTER))
        except Exception as e:
            print(f"[DEBUG] Percentile calc failed for {m}: {e}")
            percentile_df_chart[m] = 50.0

percentile_df_chart = percentile_df_chart.fillna(50.0).round(1)

# --- Percentiles for score baseline (whole dataset by position) ---
pos_col = "Six-Group Position"
if pos_col not in df_all.columns:
    df_all[pos_col] = np.nan
if pos_col not in df.columns:
    df[pos_col] = np.nan

percentile_df_globalpos_all = pd.DataFrame(index=df_all.index, columns=metrics, dtype=float)
for m in metrics:
    try:
        percentile_df_globalpos_all[m] = (
            df_all.groupby(pos_col, group_keys=False)[m]
                  .apply(lambda s: pct_rank(s, lower_is_better=(m in LOWER_IS_BETTER)))
        )
    except Exception as e:
        print(f"[DEBUG] Global percentile calc failed for {m}: {e}")
        percentile_df_globalpos_all[m] = 50.0
percentile_df_globalpos = percentile_df_globalpos_all.loc[df.index, metrics].fillna(50.0).round(1)

# --- Assemble plot_data (radar uses CHART percentiles) ---
metrics_df = df[metrics].copy()
keep_cols = [
    "Player", "Team within selected timeframe", "Team", "Age", "Height",
    "Positions played", "Minutes played", "Six-Group Position",
    "Competition", "Competition_norm", "Multiplier", "Season", "Season_norm", "PlayerKey"
]
for c in keep_cols:
    if c not in df.columns:
        df[c] = np.nan

plot_data = pd.concat(
    [df[keep_cols], metrics_df, percentile_df_chart.add_suffix(" (percentile)")],
    axis=1
)

# ============================================================
# WEIGHTED Z, SCORES AND ANCHORS
# ============================================================
sel_metrics = list(metric_groups.keys())

# Ensure metrics numeric in df_all
for m in sel_metrics:
    df_all[m] = pd.to_numeric(df_all.get(m, 0), errors="coerce").fillna(0)

# --- Raw Z per metric, by position baseline (eligible players only) ---
_mins_all = pd.to_numeric(df_all.get("Minutes played", np.nan), errors="coerce")
eligible = df_all[_mins_all >= 600].copy()
if eligible.empty:
    eligible = df_all.copy()

baseline_stats = eligible.groupby(pos_col)[sel_metrics].agg(["mean", "std"]).fillna(0)
baseline_stats.columns = baseline_stats.columns.map("_".join)

raw_z_all = pd.DataFrame(index=df_all.index, columns=sel_metrics, dtype=float)
for m in sel_metrics:
    df_all[m] = pd.to_numeric(df_all[m], errors="coerce").fillna(0)
    mean_col = f"{m}_mean"
    std_col = f"{m}_std"

    if mean_col not in baseline_stats.columns or std_col not in baseline_stats.columns:
        raw_z_all[m] = 0
        continue

    mean_vals = df_all[pos_col].map(baseline_stats[mean_col])
    std_vals = df_all[pos_col].map(baseline_stats[std_col].replace(0, 1))
    z = (df_all[m] - mean_vals) / std_vals

    if m in LOWER_IS_BETTER:
        z *= -1
    raw_z_all[m] = z.fillna(0)

# --- Average + Weighted Z ---
avg_z_all = raw_z_all.mean(axis=1).fillna(0)
df_all["Avg Z Score"] = avg_z_all

mult = pd.to_numeric(df_all.get("Multiplier", 1.0), errors="coerce").fillna(1.0)
avg_z = df_all["Avg Z Score"]

df_all["Weighted Z Score"] = np.select(
    [avg_z > 0, avg_z < 0],
    [avg_z * mult, avg_z / mult],
    default=0.0
)

# LFC Weighted version
df_all["LFC Multiplier"] = mult
df_all.loc[df_all["Competition_norm"] == "Scotland Premiership", "LFC Multiplier"] = 1.20
lfc_mult = df_all["LFC Multiplier"]

df_all["LFC Weighted Z"] = np.select(
    [avg_z > 0, avg_z < 0],
    [avg_z * lfc_mult, avg_z / lfc_mult],
    default=0.0
)

# Anchors per position
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

# Copy back into plot_data for current filtered view
plot_data["Avg Z Score"] = df_all.loc[df.index, "Avg Z Score"].fillna(0).values
plot_data["Weighted Z Score"] = df_all.loc[df.index, "Weighted Z Score"].fillna(0).values
plot_data["LFC Weighted Z"] = df_all.loc[df.index, "LFC Weighted Z"].fillna(0).values
plot_data["Score (0–100)"] = df_all.loc[df.index, "Score (0–100)"].fillna(0).values
plot_data["LFC Score (0–100)"] = df_all.loc[df.index, "LFC Score (0–100)"].fillna(0).values

# Rank
plot_data.sort_values("Weighted Z Score", ascending=False, inplace=True, ignore_index=True)
plot_data["Rank"] = np.arange(1, len(plot_data) + 1)

# ============================================================
# 3-SEASON SUMMARY WHEN “All (3-Season Avg)” IS SELECTED
# ============================================================
key_col = "PlayerKey" if "PlayerKey" in df_all.columns else "Player"

if (
    "Weighted Z Score" in df_all.columns
    and "Season" in df_all.columns
    and st.session_state.get("selected_season") == "All (3-Season Avg)"
):
    try:
        st.markdown("### 🧾 3-Season Weighted Z Summary")

        season_groups = (
            df_all.groupby([key_col, "Player", "Season", "Six-Group Position", "Competition_norm"], as_index=False)
            .agg({
                "Weighted Z Score": "mean",
                "Minutes played": "sum",
                "Multiplier": "mean",
                "Age": "mean",
                "Team": lambda x: ", ".join(sorted(set(x.dropna())))
            })
        )

        # Average across all seasons per player-position
        rolling = (
            season_groups.groupby([key_col, "Player", "Six-Group Position"], as_index=False)
            .agg({
                "Weighted Z Score": "mean",
                "Minutes played": "sum",
                "Multiplier": "mean",
                "Age": "mean",
                "Team": lambda x: ", ".join(sorted(set(x.dropna()))),
                "Competition_norm": lambda x: ", ".join(sorted(set(x.dropna())))
            })
            .rename(columns={"Weighted Z Score": "3-Season Weighted Z"})
        )

        # Improvement / trend (first vs last)
        trend_calc = (
            season_groups.sort_values([key_col, "Season"])
            .groupby(key_col)
            .agg(first_z=("Weighted Z Score", "first"), last_z=("Weighted Z Score", "last"))
        )
        trend_calc["Z Δ"] = trend_calc["last_z"] - trend_calc["first_z"]
        trend_calc["Trend"] = np.select(
            [trend_calc["Z Δ"] > 0.25, trend_calc["Z Δ"] < -0.25],
            ["⬆️ Improving", "⬇️ Declining"],
            default="➖ Stable"
        )

        rolling = rolling.merge(trend_calc, left_on=key_col, right_index=True, how="left")

        st.dataframe(
            rolling[["Player", "Six-Group Position", "3-Season Weighted Z", "Z Δ", "Trend"]]
            .sort_values("3-Season Weighted Z", ascending=False)
            .reset_index(drop=True),
            use_container_width=True
        )

        if st.checkbox("📈 Show Player Trend Over Seasons"):
            player_opts = sorted(season_groups["Player"].unique().tolist())
            player_sel = st.selectbox("Select player for trend chart", player_opts)
            p_df = season_groups[season_groups["Player"] == player_sel].sort_values("Season")

            if not p_df.empty:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(p_df["Season"], p_df["Weighted Z Score"], marker="o", color="royalblue", linewidth=2)
                ax.axhline(0, color="gray", lw=1)
                ax.set_ylabel("Weighted Z Score")
                ax.set_title(f"{player_sel} — League-Weighted Z Trend")
                st.pyplot(fig)

    except Exception as e:
        st.warning(f"⚠️ Could not compute 3-season summary: {e}")

# ============================================================
# RADAR PLOT
# ============================================================
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

    row_df = plot_data.loc[plot_data["Player"] == player_name]
    if row_df.empty:
        st.error(f"No player named '{player_name}' found.")
        return

    row = row_df.iloc[0]

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

    raw_vals = pd.to_numeric(row[valid_metrics], errors="coerce").fillna(0).to_numpy()
    pct_vals = pd.to_numeric(row[valid_pcts], errors="coerce").fillna(50).to_numpy()

    n = len(valid_metrics)
    if n == 0:
        st.warning("No valid numeric metrics found for this player.")
        return

    groups = [metric_groups.get(m, "Unknown") for m in valid_metrics]
    cmap = mcm.get_cmap("RdYlGn")
    norm = mcolors.Normalize(vmin=0, vmax=100)
    bar_colors = [cmap(norm(v)) for v in pct_vals]

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)

    ax.bar(
        angles, pct_vals,
        width=2 * np.pi / n * 0.85,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9
    )

    for ang, raw_val in zip(angles, raw_vals):
        txt = f"{raw_val:.2f}" if np.isfinite(raw_val) else "-"
        ax.text(ang, 50, txt, ha="center", va="center", color="black", fontsize=10, fontweight="bold")

    for ang, m in zip(angles, valid_metrics):
        label = DISPLAY_NAMES.get(m, m)
        label = label.replace(" per 90", "").replace(", %", " (%)")
        color = group_colors.get(metric_groups.get(m, "Unknown"), "black")
        ax.text(ang, 108, label, ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    present_groups = list(dict.fromkeys(groups))
    patches = [mpatches.Patch(color=group_colors.get(g, "grey"), label=g) for g in present_groups]
    if patches:
        fig.subplots_adjust(top=0.86, bottom=0.08)
        ax.legend(
            handles=patches,
            loc="upper center", bbox_to_anchor=(0.5, -0.06),
            ncol=min(len(patches), 4), frameon=False
        )

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

    st.pyplot(fig, width="stretch")

# ---------- Plot radar for selected player ----------
if st.session_state.selected_player:
    plot_radial_bar_grouped(
        st.session_state.selected_player,
        plot_data,
        metric_groups,
        group_colors
    )

# ============================================================
# PLAYER TREND LINE (PER PLAYER)
# ============================================================
if st.session_state.selected_player and "Season" in df_all.columns:
    p_df = (
        df_all[df_all["Player"] == st.session_state.selected_player]
        .groupby("Season", as_index=False)
        .agg({"Weighted Z Score": "mean"})
        .sort_values("Season")
    )

    if not p_df.empty and len(p_df) > 1:
        st.markdown("#### 📈 Weighted Z Score Trend by Season")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(p_df["Season"], p_df["Weighted Z Score"], marker="o", color="royalblue", linewidth=2)
        ax.axhline(0, color="gray", lw=1)
        ax.set_ylabel("Weighted Z Score")
        ax.set_xlabel("Season")
        ax.set_title(f"{st.session_state.selected_player} — Weighted Z Score by Season")
        st.pyplot(fig)

# ============================================================
# FIND SIMILAR PLAYERS
# ============================================================
st.markdown("### 🔍 Find Similar Players")

MINUTES_SIMILAR = 400
pos_col = "Six-Group Position"

def find_similar_players_same_scale(
    player_name: str,
    base_df: pd.DataFrame,
    metrics: list,
    position_col: str,
    lower_is_better: set,
    n_similar: int = 10,
    minutes_floor: int = 400
):
    if player_name not in base_df["Player"].values:
        return pd.DataFrame(), f"{player_name} not found."

    pos = base_df.loc[base_df["Player"] == player_name, position_col].iloc[0]
    if pd.isna(pos) or not pos:
        return pd.DataFrame(), f"No position found for {player_name}."

    mins_num = pd.to_numeric(base_df.get("Minutes played", np.nan), errors="coerce")
    pool = base_df[
        (base_df[position_col] == pos) &
        (mins_num >= minutes_floor)
    ].copy()
    if pool.empty:
        return pd.DataFrame(), f"No players (≥{minutes_floor} mins) at {pos}."

    valid_metrics = [m for m in metrics if m in pool.columns]
    if not valid_metrics:
        return pd.DataFrame(), "No valid metrics for similarity."

    X = pool[valid_metrics].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    Z = (X - X.mean()) / X.std(ddof=0).replace(0, np.nan)
    Z = Z.fillna(0)

    pZ = Z.loc[pool["Player"] == player_name]
    if pZ.empty:
        return pd.DataFrame(), f"{player_name} has no valid metric data."

    diff = Z.values - pZ.values[0]
    dists = np.sqrt(np.sum(diff**2, axis=1))
    pool["Similarity Score"] = 100.0 - (dists / dists.max() * 100.0 if dists.max() != 0 else 0)

    out = (
        pool[pool["Player"] != player_name]
        .sort_values("Similarity Score", ascending=False)
        .head(n_similar)
        .copy()
    )

    score_map = base_df.set_index("Player")["Score (0–100)"].to_dict()
    out["Score (0–100)"] = out["Player"].map(score_map)

    out.rename(columns={"Team within selected timeframe": "Team",
                        "Competition_norm": "League"}, inplace=True)
    keep_cols = ["Player", "Team", "League", "Age", "Minutes played", "Score (0–100)", "Similarity Score"]
    for c in keep_cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[keep_cols]

    out["Minutes played"] = pd.to_numeric(out["Minutes played"], errors="coerce").fillna(0).astype(int)
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce").round(0)
    out["Score (0–100)"] = pd.to_numeric(out["Score (0–100)"], errors="coerce").round(1)
    out["Similarity Score"] = pd.to_numeric(out["Similarity Score"], errors="coerce").round(1)

    return out, None

if st.button("Find 10 Similar Players", key="similar_players_button"):
    with st.spinner("Finding most similar players..."):
        similar_df, err = find_similar_players_same_scale(
            st.session_state.selected_player,
            plot_data,
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
        similar_df = similar_df.loc[:, ~similar_df.columns.duplicated()]
        similar_df = similar_df[["Player", "Team", "League", "Age", "Minutes played", "Score (0–100)", "Similarity Score"]]
        st.dataframe(similar_df, use_container_width=True)

# ============================================================
# PLAYER TREND ANALYSIS & IMPROVEMENT TAGS (GLOBAL)
# ============================================================
if "Season" in df_all.columns and "Weighted Z Score" in df_all.columns:
    season_groups = (
        df_all.groupby([key_col, "Player", "Season"], as_index=False)
        .agg({"Weighted Z Score": "mean", "Minutes played": "sum"})
    )

    trend_calc = (
        season_groups.sort_values([key_col, "Season"])
        .groupby(key_col)
        .agg(first_z=("Weighted Z Score", "first"), last_z=("Weighted Z Score", "last"))
    )
    trend_calc["ΔZ"] = trend_calc["last_z"] - trend_calc["first_z"]

    trend_calc["Trend"] = np.select(
        [
            trend_calc["ΔZ"] >= 0.5,
            (trend_calc["ΔZ"] >= 0.2) & (trend_calc["ΔZ"] < 0.5),
            (trend_calc["ΔZ"] <= -0.2) & (trend_calc["ΔZ"] > -0.5),
            trend_calc["ΔZ"] <= -0.5,
        ],
        ["⬆️ Improving a lot", "↗️ Improving", "↘️ Declining", "⬇️ Declining a lot"],
        default="➖ Stable",
    )

    df_all = df_all.merge(trend_calc[["Trend"]], left_on=key_col, right_index=True, how="left")
else:
    df_all["Trend"] = "Unknown"

# ============================================================
# RANKING TABLE + FAVOURITES INTEGRATION
# ============================================================
st.markdown("### Players Ranked by Score (0–100)")

cols_for_table = [
    "Player", "Positions played", "Team", "Competition_norm", "Multiplier",
    "Avg Z Score", "Weighted Z Score", "Score (0–100)", "LFC Score (0–100)",
    "Trend", "Age", "Minutes played", "Rank", "Season", "Season_norm"
]

for c in cols_for_table:
    if c not in plot_data.columns:
        plot_data[c] = df_all.loc[plot_data.index, c] if c in df_all.columns else np.nan

z_ranking = plot_data[cols_for_table].copy()

if "Season" in z_ranking.columns and "selected_season" in st.session_state and st.session_state["selected_season"] != "All (3-Season Avg)":
    sel_season = st.session_state["selected_season"]
    z_ranking = (
        z_ranking.assign(
            _is_selected=z_ranking["Season"].astype(str).eq(sel_season)
        )
        .sort_values(["_is_selected", "Weighted Z Score"], ascending=[False, False])
        .drop_duplicates(subset=["Player"], keep="first")
        .drop(columns="_is_selected")
    )
else:
    z_ranking = z_ranking.sort_values("Weighted Z Score", ascending=False)
    z_ranking = z_ranking.drop_duplicates(subset=["Player"], keep="first")

z_ranking = z_ranking.sort_values("Weighted Z Score", ascending=False, ignore_index=True)
z_ranking["Rank"] = np.arange(1, len(z_ranking) + 1)

if "Competition_norm" in z_ranking.columns:
    z_ranking.rename(columns={"Competition_norm": "League"}, inplace=True)

# Favourites from Supabase
from lib.favourites_repo import get_supabase_client

def get_favourites_with_colours_live():
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

favs = get_favourites_with_colours_live()

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
    data = favs_dict.get(name)
    if not data:
        return name
    emoji = COLOUR_EMOJI.get(str(data.get("colour", "")).strip(), "")
    return f"{emoji} {name}" if emoji else name

z_ranking["Player (coloured)"] = z_ranking["Player"].apply(lambda n: colourize_player_name(n, favs))
z_ranking["⭐ Favourite"] = z_ranking["Player"].apply(lambda n: bool(favs.get(n, {}).get("visible", False)))

required_cols = [
    "⭐ Favourite", "Player (coloured)", "Positions played", "Team", "League",
    "Multiplier", "Avg Z Score", "Weighted Z Score",
    "Score (0–100)", "LFC Score (0–100)",
    "Trend", "Age", "Minutes played", "Rank"
]
for col in required_cols:
    if col not in z_ranking.columns:
        z_ranking[col] = np.nan
z_ranking = z_ranking[required_cols]

# Editable table
sig_parts = (
    tuple(sorted(selected_leagues)),
    int(min_minutes),
    tuple(selected_groups),
    selected_position_template,
    len(z_ranking),
    float(z_ranking["Score (0–100)"].sum())
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
# SYNC FAVOURITES TO SUPABASE
# ============================================================
import time

@st.cache_data(ttl=5, show_spinner=False)
def load_favourites_cached():
    return get_favourites_with_colours_live()

favs_live = load_favourites_cached()

if st.session_state.get("_last_sync_time") and time.time() - st.session_state["_last_sync_time"] < 3:
    pass
else:
    st.session_state["_last_sync_time"] = time.time()

    if "⭐ Favourite" not in edited_df.columns:
        st.warning("⚠️ Could not find the '⭐ Favourite' column — skipping sync.")
    else:
        favourite_rows = edited_df[edited_df["⭐ Favourite"] == True].copy()
        deleted_players = {p for p, d in favs_live.items() if not d.get("visible", True)}

        for _, row in favourite_rows.iterrows():
            player_raw = str(row.get("Player (coloured)", "")).strip()
            player_name = re.sub(r"^[🟢🟡🔴🟣🟠]\s*", "", player_raw).strip()
            team = row.get("Team", "")
            league = row.get("League", "")
            position = row.get("Positions played", "")

            prev_data = favs_live.get(player_name, {})
            prev_visible = bool(prev_data.get("visible", False))

            if player_name in deleted_players and not prev_visible:
                continue

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
                    "source": "radar-test-page",
                }

                upsert_favourite(payload, log_to_sheet=True)
            else:
                pass

        non_fav_rows = edited_df[edited_df["⭐ Favourite"] == False]
        for _, row in non_fav_rows.iterrows():
            player_raw = str(row.get("Player (coloured)", "")).strip()
            player_name = re.sub(r"^[🟢🟡🔴🟣🟠]\s*", "", player_raw).strip()

            old_visible = favs_live.get(player_name, {}).get("visible", False)
            if old_visible:
                hide_favourite(player_name)
