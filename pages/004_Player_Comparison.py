import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from datetime import datetime
from auth import check_password
from branding import show_branding
from ui.sidebar import render_sidebar
from openai import OpenAI
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(page_title="Livingston FC Recruitment App", layout="centered")


# ---------- Authentication ----------
if not check_password():
    st.stop()
render_sidebar()

# ---------- Branding ----------
show_branding()
st.title("Player Comparison")

# ---------- Fixed group colours ----------
group_colors = {
    "Attacking":   "crimson",
    "Possession":  "seagreen",
    "Defensive":   "royalblue",
}
GENRE_ALPHA = 0.08

# ========== Position metrics ==========
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
    "Australia A-League": "Australia A-League Men",
    "Australia A-League Men": "Australia A-League Men",
    "A-League": "Australia A-League Men",
    "Australia A-League": "Australia A-League Men",
    "Australia A-League": "A-League",
    "A-League": "Australia A-League",
    "2. Liga": "Austria 2. Liga",
    "Challenger Pro League": "Belgium Challenger Pro League",
    "Belgium Challenger Pro League": "Belgium Challenger Pro League",
    "Jupiler Pro League": "Jupiler Pro League",
    "Belgium Pro League": "Jupiler Pro League",
    "Belgian Pro League": "Jupiler Pro League",
    "Belgium Jupiler Pro League": "Jupiler Pro League",
    "First League": "Bulgaria First League",
    "Bulgaria First League": "Bulgaria First League",
    "1. HNL": "Croatia 1. HNL",
    "HNL": "Croatia 1. HNL",
    "Croatia 1. HNL": "Croatia 1. HNL",
    "Czech Liga": "Czech First Tier",
    "Czech First Tier": "Czech First Tier",
    "1st Division": "Denmark 1st Division",
    "Denmark 1st Division": "Denmark 1st Division",
    "Superliga": "Denmark Superliga",
    "Denmark Superliga": "Denmark Superliga",
    "League One": "England League One",
    "England League One": "England League One",
    "League Two": "England League Two",
    "England League Two": "England League Two",
    "National League": "England National League",
    "England National League": "England National League",
    "National League N / S": "England National League N/S",
    "England National League N/S": "England National League N/S",
    "Premium Liiga": "Estonia Premium Liiga",
    "Estonia Premium Liiga": "Estonia Premium Liiga",
    "Veikkausliiga": "Finland Veikkausliiga",
    "Finland Veikkausliiga": "Finland Veikkausliiga",
    "Championnat National": "France National 1",
    "France National 1": "France National 1",
    "Ligue 2": "Ligue 2",
    "France Ligue 2": "Ligue 2",
    "2. Bundesliga": "2. Bundesliga",
    "Germany 2. Bundesliga": "2. Bundesliga",
    "3. Liga": "Germany 3. Liga",
    "Germany 3. Liga": "Germany 3. Liga",
    "Super League": "Greece Super League 1",
    "Greece Super League": "Greece Super League 1",
    "Greece Super League 1": "Greece Super League 1",
    "NB I": "Hungary NB I",
    "Hungary NB I": "Hungary NB I",
    "Besta deild karla": "Iceland Besta Deild",
    "Iceland Besta Deild": "Iceland Besta Deild",
    "Serie C": "Italy Serie C",
    "Italy Serie C": "Italy Serie C",
    "J2 League": "Japan J2 League",
    "Japan J2 League": "Japan J2 League",
    "Virsliga": "Latvia Virsliga",
    "Latvia Virsliga": "Latvia Virsliga",
    "A Lyga": "Lithuania A Lyga",
    "Lithuania A Lyga": "Lithuania A Lyga",
    "Botola Pro": "Morocco Botola Pro",
    "Morocco Botola Pro": "Morocco Botola Pro",
    "Eredivisie": "Eredivisie",
    "Netherlands Eredivisie": "Eredivisie",
    "Eerste Divisie": "Netherlands Eerste Divisie",
    "Netherlands Eerste Divisie": "Netherlands Eerste Divisie",
    "1. Division": "Norway 1. Division",
    "Norway 1. Division": "Norway 1. Division",
    "Eliteserien": "Norway Eliteserien",
    "Norway Eliteserien": "Norway Eliteserien",
    "I Liga": "Poland 1 Liga",
    "Poland 1 Liga": "Poland 1 Liga",
    "Ekstraklasa": "Poland Ekstraklasa",
    "Poland Ekstraklasa": "Poland Ekstraklasa",
    "Segunda Liga": "Portugal Segunda Liga",
    "Liga Pro": "Portugal Segunda Liga",
    "Portugal Segunda Liga": "Portugal Segunda Liga",
    "Premier Division": "Republic of Ireland Premier Division",
    "Ireland Premier Division": "Republic of Ireland Premier Division",
    "Republic of Ireland Premier Division": "Republic of Ireland Premier Division",
    "Liga 1": "Romania Liga 1",
    "Romania Liga 1": "Romania Liga 1",
    "Championship": "Scotland Championship",
    "Scotland Championship": "Scotland Championship",
    "Premiership": "Scotland Premiership",
    "Scotland Premiership": "Scotland Premiership",
    "Super Liga": "Serbia Super Liga",
    "Serbia Super Liga": "Serbia Super Liga",
    "Slovakia Super Liga": "Slovakia 1. Liga",
    "Slovakia First League": "Slovakia 1. Liga",
    "1. Liga": "Slovakia 1. Liga",
    "Slovakia 1. Liga": "Slovakia 1. Liga",
    "1. Liga (SVN)": "Slovenia 1. Liga",
    "Slovenia 1. SNL": "Slovenia 1. Liga",
    "Slovenia 1. Liga": "Slovenia 1. Liga",
    "PSL": "South Africa Premier Division",
    "South Africa Premier Division": "South Africa Premier Division",
    "Allsvenskan": "Sweden Allsvenskan",
    "Sweden Allsvenskan": "Sweden Allsvenskan",
    "Superettan": "Sweden Superettan",
    "Sweden Superettan": "Sweden Superettan",
    "Challenge League": "Switzerland Challenge League",
    "Switzerland Challenge League": "Switzerland Challenge League",
    "Denmark 1. Division": "Denmark 1st Division",

    # --- Tunisia fixes ---
    "Ligue 1": "Tunisia Ligue 1",      # bare 'Ligue 1' should always mean Tunisia
    "Ligue 1 (TUN)": "Tunisia Ligue 1",
    "Tunisia Ligue 1": "Tunisia Ligue 1",
    "France Ligue 1": "Tunisia Ligue 1",

    # --- USA ---
    "USL Championship": "USA USL Championship",
    "USA USL Championship": "USA USL Championship",
}

# ========== Role groups shown in filters ==========
SIX_GROUPS = ["Full Back", "Centre Back", "Number 6", "Number 8", "Number 10", "Winger", "Striker"]

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
    "DEFENSIVEMIDFIELDER": "Number 6",
    "RIGHTDEFENSIVEMIDFIELDER": "Number 6",
    "LEFTDEFENSIVEMIDFIELDER": "Number 6",
    "CENTREDEFENSIVEMIDFIELDER": "Number 6",

    # Number 8
    "CENTREBOXTOBOXMIDFIELDER": "Number 8",
    "BOXTOBOXMIDFIELDER": "Number 8",
    "RIGHTBOXTOBOXMIDFIELDER": "Number 8",
    "LEFTBOXTOBOXMIDFIELDER": "Number 8",
    "NUMBER8": "Number 8",

    # Number 10
    "CENTREATTACKINGMIDFIELDER": "Number 10",
    "ATTACKINGMIDFIELDER": "Number 10",
    "SECONDSTRIKER": "Number 10",
    "10": "Number 10",
    "NUMBER10": "Number 10",
    "RIGHTATTACKINGMIDFIELDER": "Number 10",
    "LEFTATTACKINGMIDFIELDER": "Number 10",

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
    "Number 10": "Number 10",
    "Winger": "Winger",
    "Striker": "Striker"
}

# ========== Radar metric sets ==========
position_metrics = {
    # ---------- Centre Back ----------
    "Centre Back": {
        "metrics": [
            "NP Goals",
            "Passing%", "Pass OBV", "Pr. Long Balls", "UPr. Long Balls", "OBV", "Pr. Pass% Dif.",
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
            "xGBuildup", "xG Assisted",
            "Passing%", "Deep Progressions", "Turnovers", "OBV", "Pass OBV", "Pr. Pass% Dif.",
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
            "xGBuildup", "xG Assisted", "Shots", "xG", "NP Goals",
            "Passing%", "Deep Progressions", "OP Passes Into Box", "Pass OBV", "OBV", "Deep Completions",
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
    # ---------- Number 10 ----------
    "Number 10": {
        "metrics": [
            "Deep Progressions", "Deep Completions", "OP Passes Into Box", "OBV", "OP Key Passes",
            "Shots", "xG", "xG Assisted", "Assists", "NP Goals", "xG/Shot",
            "Touches In Box", "Goal Conversion%",
            "Aggressive Actions", "PAdj Pressures",
        ],
        "groups": {
            "Deep Progressions": "Possession",
            "Deep Completions": "Possession",
            "OP Passes Into Box": "Possession",
            "OP Key Passes": "Possession",
            "OBV": "Possession",
            "Shots": "Attacking",
            "xG": "Attacking",
            "xG Assisted": "Attacking",
            "Assists": "Attacking",
            "NP Goals": "Attacking",
            "xG/Shot": "Attacking",
            "Touches In Box": "Attacking",
            "Goal Conversion%": "Attacking",
            "Aggressive Actions": "Defensive",
            "PAdj Pressures": "Defensive",
        }
    },
    # ---------- Winger ----------
    "Winger": {
        "metrics": [
            "xG", "Shots", "xG/Shot", "Touches In Box", "OP xG Assisted", "NP Goals",
            "OP Passes Into Box", "Successful Box Cross%", "Passing%",
            "Successful Dribbles", "Turnovers", "OBV", "D&C OBV", "Fouls Won", "Deep Progressions",
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

def load_one_file(p: Path) -> pd.DataFrame:
    print(f"[DEBUG] Trying to load file at: {p.resolve()}")

    def try_excel() -> pd.DataFrame | None:
        try:
            import openpyxl
            return pd.read_excel(p, engine="openpyxl")
        except ImportError:
            print("[DEBUG] openpyxl not available, trying CSV reader next.")
            return None
        except Exception as e:
            print(f"[DEBUG] Excel read failed: {e}. Trying CSV instead.")
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

    # Decide reader
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
        raise ValueError(f"Unsupported or unreadable file: {p.name}")

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

def add_age_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add numeric Age column based on birth_date."""
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

@st.cache_data(show_spinner=False)
def load_statsbomb(path: Path, _sig=None) -> pd.DataFrame:
    print(f"[DEBUG] Data path configured as: {path}")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Put a CSV or XLSX there, or a folder of them.")

    if path.is_file():
        df = load_one_file(path)
    else:
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
        print(f"[DEBUG] Merged {len(files)} files, total rows {len(df)}")

    # Always add Age column
    df = add_age_column(df)
    return df

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
        multipliers_df = pd.read_excel("league_multipliers.xlsx")
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
    if "Name" in df.columns: rename_map["Name"] = "Player"
    if "Primary Position" in df.columns: rename_map["Primary Position"] = "Position"
    if "Minutes" in df.columns: rename_map["Minutes"] = "Minutes played"

    # --- Metric renames / fixes ---
    rename_map.update({
        # Successful Box Cross variants
        "Successful Box Cross %": "Successful Box Cross%",
        "Player Season Box Cross Ratio": "Successful Box Cross%",

        # Pass% under pressure
        "Player Season Change In Passing Ratio": "Pr. Pass% Dif.",

        # Build-up involvement
        "Player Season Xgbuildup 90": "xGBuildup",

        # Pressures in attacking 3rd
        "Player Season F3 Pressures 90": "Pressures in Final 1/3",

        # Long balls
        "Player Season Pressured Long Balls 90": "Pr. Long Balls",
        "Player Season Unpressured Long Balls 90": "UPr. Long Balls",
    })

    df.rename(columns=rename_map, inplace=True)

    # --- Derive Successful Crosses ---
    if "Crosses" in df.columns and "Crossing%" in df.columns:
        df["Successful Crosses"] = (
            pd.to_numeric(df["Crosses"], errors="coerce") *
            (pd.to_numeric(df["Crossing%"], errors="coerce") / 100.0)
        )

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

    # --- Six-Group Position mapping (primary + secondary, fully aligned with Historical Radar) ---
    def map_positions_to_groups(row):
        groups = set()

        primary = map_first_position_to_group(row.get("Position", None))
        secondary = map_first_position_to_group(row.get("Secondary Position", None))

        if primary:
            groups.add(primary)
        if secondary:
            groups.add(secondary)

        # Expand generic Centre Midfield into both 6 & 8
        if "Centre Midfield" in groups:
            groups.remove("Centre Midfield")
            groups.update(["Number 6", "Number 8"])

        # Expand attacking midfielders into Number 8 as well
        if "Number 10" in groups:
            groups.add("Number 8")

        return list(groups)

    if "Position" in df.columns:
        df["_six_groups_list"] = df.apply(map_positions_to_groups, axis=1)
        df = df.explode("_six_groups_list")
        df["Six-Group Position"] = df["_six_groups_list"]
        df.drop(columns=["_six_groups_list"], inplace=True, errors="ignore")
        # After exploding positions, ensure a clean, unique index
        df = df.reset_index(drop=True)
    else:
        df["Six-Group Position"] = np.nan

    return df


# ---------- Data source: historical StatsBomb season files ----------
ROOT_DIR = Path(__file__).parent.parent
HIST_DIR = ROOT_DIR / "data" / "statsbomb"

@st.cache_data(show_spinner=False)
def load_all_historical_statsbomb(base_dir: Path) -> pd.DataFrame:
    frames = []

    for league_dir in sorted(base_dir.iterdir()):
        if not league_dir.is_dir():
            continue

        league_name = league_dir.name

        for f in league_dir.glob("*_clean.csv"):
            season = f.stem.replace("_clean", "")

            try:
                df_season = pd.read_csv(f)
            except Exception:
                continue

            df_season["League"] = league_name
            df_season["Season"] = season
            frames.append(df_season)

    if not frames:
        raise ValueError("No historical StatsBomb files found in data/statsbomb")

    df_all = pd.concat(frames, ignore_index=True, sort=False)
    return df_all

# ---------- Load + preprocess historical data ----------
df_all_raw = load_all_historical_statsbomb(HIST_DIR).copy()

df_all_raw.columns = (
    df_all_raw.columns.astype(str)
    .str.strip()
    .str.replace(u"\xa0", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
)

# Add Age column from Birth Date if available
if "Birth Date" in df_all_raw.columns:
    today = datetime.today()
    df_all_raw["Age"] = pd.to_datetime(
        df_all_raw["Birth Date"], errors="coerce"
    ).apply(
        lambda dob: today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        if pd.notna(dob) else np.nan
    )

df_all = preprocess_df(df_all_raw)
df = df_all.copy()

# ---------- League filter ----------
league_col = "Competition_norm" if "Competition_norm" in df.columns else "Competition"
df[league_col] = df[league_col].astype(str).str.strip()

# Collect all available leagues from the dataset
all_leagues = sorted([x for x in df[league_col].dropna().unique() if x != ""])

st.markdown("#### Choose league(s)")

# --- Ensure session state defaults are valid ---
if "league_selection" not in st.session_state:
    # Initialise with all leagues selected
    st.session_state.league_selection = all_leagues.copy()
else:
    # Remove any invalid or outdated leagues from the session state
    st.session_state.league_selection = [
        l for l in st.session_state.league_selection if l in all_leagues
    ]

# --- Buttons for quick select/clear ---
b1, b2, _ = st.columns([1, 1, 6])
with b1:
    if st.button("Select all"):
        st.session_state.league_selection = all_leagues.copy()
with b2:
    if st.button("Clear all"):
        st.session_state.league_selection = []

# --- Multiselect control ---
selected_leagues = st.multiselect(
    "Leagues",
    options=all_leagues,
    default=st.session_state.league_selection,
    key="league_selection",
    label_visibility="collapsed"
)

# --- Apply filter or stop if empty ---
if selected_leagues:
    df = df[df[league_col].isin(selected_leagues)].copy()
    st.caption(f"Leagues selected: {len(selected_leagues)} | Players: {len(df)}")
    if df.empty:
        st.warning("No players match these leagues.")
        st.stop()
else:
    st.warning("Please select at least one league.")
    st.stop()

# ---------- Minutes + Age filters ----------
minutes_col = "Minutes played"
if minutes_col not in df.columns:
    df[minutes_col] = np.nan

c1, c2 = st.columns(2)
with c1:
    if "min_minutes_cmp" not in st.session_state:
        st.session_state.min_minutes_cmp = 1000
    st.session_state.min_minutes_cmp = st.number_input(
        "Minimum minutes",
        min_value=0,
        value=st.session_state.min_minutes_cmp,
        step=50,
        key="min_minutes_cmp_input"
    )
    min_minutes = st.session_state.min_minutes_cmp
    df["_minutes_numeric"] = pd.to_numeric(df[minutes_col], errors="coerce")
    df = df[df["_minutes_numeric"] >= min_minutes].copy()

with c2:
    if "Age" in df.columns:
        df["_age_numeric"] = pd.to_numeric(df["Age"], errors="coerce")
        if df["_age_numeric"].notna().any():
            age_min = int(np.nanmin(df["_age_numeric"]))
            age_max = int(np.nanmax(df["_age_numeric"]))
            if "age_range_cmp" not in st.session_state:
                st.session_state.age_range_cmp = (age_min, age_max)
            sel_min, sel_max = st.slider(
                "Age range",
                min_value=age_min,
                max_value=age_max,
                value=st.session_state.age_range_cmp,
                step=1,
                key="age_range_cmp_slider"
            )
            st.session_state.age_range_cmp = (sel_min, sel_max)
            df = df[df["_age_numeric"].between(sel_min, sel_max)].copy()

st.caption(f"Filtering on '{minutes_col}' ≥ {min_minutes}. Players remaining {len(df)}")

# ---------- Position group ----------
st.markdown("#### 🟡 Select Position Group")

available_groups = [
    g for g in SIX_GROUPS
    if "Six-Group Position" in df.columns and g in df["Six-Group Position"].unique()
]

if "selected_groups_cmp" not in st.session_state:
    st.session_state.selected_groups_cmp = []

selected_groups = st.multiselect(
    "Position Groups",
    options=available_groups,
    default=st.session_state.selected_groups_cmp,
    key="pos_group_multiselect_cmp",
    label_visibility="collapsed"
)
st.session_state.selected_groups_cmp = selected_groups

if selected_groups:
    df = df[df["Six-Group Position"].isin(selected_groups)].copy()
if df.empty:
    st.stop()

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
    "Radar Template",
    template_names,
    index=template_names.index(st.session_state.template_select),
    key="template_select",
    label_visibility="collapsed"
)

# Handle manual override
if st.session_state.template_select != st.session_state.last_template_choice:
    st.session_state.manual_override = True
    st.session_state.last_template_choice = st.session_state.template_select

# ---------- Metrics for radar ----------
current_template_name = st.session_state.template_select or list(position_metrics.keys())[0]
metrics = position_metrics[current_template_name]["metrics"]
# Force metric order to exactly match StatsBomb Radar templates
metrics = list(position_metrics[current_template_name]["metrics"])
metric_groups = position_metrics[current_template_name]["groups"]

# ---------- Season handling ----------
SEASON_COL_CANDIDATES = ["Season", "season", "Season Name", "season_name"]
season_col = next((c for c in SEASON_COL_CANDIDATES if c in df.columns), None)

if season_col:
    df[season_col] = df[season_col].astype(str).str.strip()
else:
    season_col = None

# ---------- Player A & B (with season selection) ----------
players = df["Player"].dropna().unique().tolist()
if not players:
    st.stop()

if "cmpA" not in st.session_state:
    st.session_state.cmpA = players[0]
if "cmpB" not in st.session_state:
    st.session_state.cmpB = players[1] if len(players) > 1 else None

c1, c2 = st.columns(2)

with c1:
    pA = st.selectbox(
        "Player A",
        players,
        index=players.index(st.session_state.cmpA) if st.session_state.cmpA in players else 0,
        key="cmpA_sel"
    )
    st.session_state.cmpA = pA

    if season_col:
        seasons_A = (
            df.loc[df["Player"] == pA, season_col]
            .dropna()
            .unique()
            .tolist()
        )
        seasons_A = sorted(seasons_A)

        if "seasonA" not in st.session_state or st.session_state.seasonA not in seasons_A:
            st.session_state.seasonA = seasons_A[0] if seasons_A else None

        seasonA = st.selectbox(
            "Season (Player A)",
            seasons_A,
            index=seasons_A.index(st.session_state.seasonA) if st.session_state.seasonA in seasons_A else 0,
            key="seasonA_sel"
        )
        st.session_state.seasonA = seasonA
    else:
        seasonA = None

with c2:
    pB = st.selectbox(
        "Player B (optional)",
        ["(none)"] + players,
        index=(players.index(st.session_state.cmpB) + 1) if st.session_state.cmpB in players else 0,
        key="cmpB_sel"
    )
    pB = None if pB == "(none)" else pB
    st.session_state.cmpB = pB

    if pB and season_col:
        seasons_B = (
            df.loc[df["Player"] == pB, season_col]
            .dropna()
            .unique()
            .tolist()
        )
        seasons_B = sorted(seasons_B)

        if "seasonB" not in st.session_state or st.session_state.seasonB not in seasons_B:
            st.session_state.seasonB = seasons_B[0] if seasons_B else None

        seasonB = st.selectbox(
            "Season (Player B)",
            seasons_B,
            index=seasons_B.index(st.session_state.seasonB) if st.session_state.seasonB in seasons_B else 0,
            key="seasonB_sel"
        )
        st.session_state.seasonB = seasonB
    else:
        seasonB = None

# ---------- Metrics where lower values are better (match Statsbomb Radar page) ----------
# NOTE: only percentiles are inverted, raw values are unchanged.
LOWER_IS_BETTER = {
    "Turnovers",
    "Fouls",
    "Pr. Long Balls",
    "UPr. Long Balls",
}

def pct_rank(series: pd.Series, lower_is_better: bool) -> pd.Series:
    """Match Statsbomb Radar page percentile behaviour."""
    series = pd.to_numeric(series, errors="coerce").fillna(0)
    r = series.rank(pct=True, ascending=True)
    p = (1.0 - r) if lower_is_better else r
    return (p * 100.0).round(1)


# ---------- Percentiles (match Statsbomb Radar page behaviour) ----------
# Ensure metric columns exist and are numeric
for m in metrics:
    if m not in df.columns:
        df[m] = 0
    df[m] = pd.to_numeric(df[m], errors="coerce").fillna(0)

# A) Percentiles for RADAR BARS (within selected leagues vs pooled)
league_col = "Competition_norm" if "Competition_norm" in df.columns else "Competition"
compute_within_league = st.checkbox(
    "Percentiles within each league",
    value=True,
    key="percentiles_within_league_cmp"
)

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


def get_pct_row(player, season):
    if not player:
        return None

    sub = df[df["Player"] == player]
    if season_col and season:
        sub = sub[sub[season_col] == season]

    if sub.empty:
        return None

    idx = sub.index[0]
    row = percentile_df_chart.loc[idx, metrics]

    # Ensure no duplicate metric labels before reindexing
    row = row[~row.index.duplicated(keep="first")]

    # Reindex safely to the metrics order
    return row.reindex(metrics).fillna(0)


rowA_pct = get_pct_row(pA, seasonA)
rowB_pct = get_pct_row(pB, seasonB) if pB else None

# --- Resolve single rows for title display (team / season) ---
def get_player_row_for_title(player, season):
    sub = df[df["Player"] == player]
    if season_col and season:
        sub = sub[sub[season_col] == season]
    return sub.iloc[0] if not sub.empty else None

rowA = get_player_row_for_title(pA, seasonA)
rowB = get_player_row_for_title(pB, seasonB) if pB else None

# ---------- Radar compare ----------
def radar_compare(labels, A_vals, B_vals=None, A_name="A", B_name="B",
                  A_team="", B_team="", A_season="", B_season="",
                  labels_to_genre=None, genre_colors=None):
    import matplotlib.patches as mpatches

    if not labels:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No metrics", ha="center", va="center")
        return fig

    # Defensive guard: prevent crashes if data lengths do not match labels
    if A_vals is None or len(A_vals) != len(labels):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "Radar data unavailable", ha="center", va="center")
        return fig
    if B_vals is not None and len(B_vals) != len(labels):
        B_vals = None

    # Colours to match Livingston badge
    color_A = "#C9A227"  # Gold
    color_B = "#000000"  # Black

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    A = list(A_vals) + [A_vals[0]]
    B = list(B_vals) + [B_vals[0]] if B_vals is not None else None

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.spines["polar"].set_visible(False)

    # Remove radial angle labels
    ax.set_xticks([])

    # Plot radar lines
    ax.plot(angles, A, linewidth=2.5, color=color_A, label=A_name)
    ax.fill(angles, A, color=color_A, alpha=0.25)
    if B is not None:
        ax.plot(angles, B, linewidth=2.5, color=color_B, label=B_name)
        ax.fill(angles, B, color=color_B, alpha=0.15)

    # Metric labels, coloured by group
    for ang, lbl in zip(angles[:-1], labels):
        group = labels_to_genre.get(lbl, "")
        color = genre_colors.get(group, "black") if genre_colors else "black"
        ax.text(ang, 108, lbl, ha="center", va="center",
                fontsize=10, fontweight="bold", color=color)

    # ----- Custom left/right aligned titles -----
    if B_name:
        # Player A (left, yellow)
        ax.text(
            0.02, 1.14,
            f"{A_name}\n{A_team}\n{A_season}",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=15, fontweight="bold", color=color_A
        )

        # Player B (right, black)
        ax.text(
            0.98, 1.14,
            f"{B_name}\n{B_team}\n{B_season}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=15, fontweight="bold", color=color_B
        )
    else:
        # Single-player title (left aligned)
        ax.text(
            0.02, 1.14,
            f"{A_name}\n{A_team}\n{A_season}",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=15, fontweight="bold", color=color_A
        )

    # Legend for metric groups at the bottom
    if labels_to_genre and genre_colors:
        present_groups = sorted(set(labels_to_genre.values()))
        patches = [mpatches.Patch(color=genre_colors[g], label=g)
                   for g in present_groups if g in genre_colors]
        ax.legend(handles=patches,
                  loc="upper center", bbox_to_anchor=(0.5, -0.08),
                  ncol=len(patches), frameon=False)

    return fig

labels_clean = []
for m in metrics:
    nice = DISPLAY_NAMES.get(m, m)  # use override if it exists
    nice = nice.replace(" per 90", "").replace(", %", " (%)")
    labels_clean.append(nice)
labels_to_genre = {
    lbl: metric_groups.get(m, "Other")
    for lbl, m in zip(labels_clean, metrics)
}

A_vals = rowA_pct.values if rowA_pct is not None else np.zeros(len(metrics))
B_vals = rowB_pct.values if rowB_pct is not None else None

fig = radar_compare(
    labels_clean, A_vals, B_vals,
    A_name=pA, B_name=pB,
    A_team=rowA.get("Team", "") if rowA is not None else "",
    B_team=rowB.get("Team", "") if rowB is not None else "",
    A_season=seasonA,
    B_season=seasonB,
    labels_to_genre=labels_to_genre,
    genre_colors=group_colors
)
st.pyplot(fig, width="stretch")

def generate_comparison_summary(player_a, player_b, df, metrics):
    """Generate AI comparison between two players using OpenAI."""
    # Create OpenAI client lazily so the page doesn't crash if secrets are missing
    try:
        client = OpenAI(api_key=st.secrets["OpenAI"]["OPENAI_API_KEY"])
    except Exception as e:
        return f"⚠️ AI summary unavailable (OpenAI config error): {e}"
    if not player_a or not player_b:
        return "Please select two players to compare."

    # Extract player rows
    rowA = df.loc[df["Player"] == player_a]
    rowB = df.loc[df["Player"] == player_b]
    if rowA.empty or rowB.empty:
        return "Player data missing for one or both players."

    # Collect metric info
    def player_snapshot(row):
        summary = {}
        for m in metrics:
            if m in row.columns:
                val = pd.to_numeric(row[m].values[0], errors="coerce")
                if not np.isnan(val):
                    summary[m] = round(val, 2)
        return summary

    summaryA = player_snapshot(rowA)
    summaryB = player_snapshot(rowB)

    # Construct prompt
    prompt = f"""
    You are a professional football recruitment analyst.

    Compare these two players based on the provided performance metrics and describe their relative strengths, weaknesses, and play styles. 
    Be concise, data-driven, and objective — like a scout report comparison.

    PLAYER A: {player_a}
    METRICS: {summaryA}

    PLAYER B: {player_b}
    METRICS: {summaryB}

    Structure the response as:
    - Overview
    - Strengths of {player_a}
    - Strengths of {player_b}
    - Summary (which profiles better for different tactical systems)
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ AI summary failed: {e}"

# ---------- AI Scouting Comparison ----------
st.markdown("### 🧠 AI Scouting Comparison")

if pA and pB:
    if st.button("Generate AI Comparison", key="ai_cmp_button"):
        with st.spinner("Generating AI scouting comparison..."):
            summary = generate_comparison_summary(pA, pB, df, metrics)
            st.markdown(summary)
else:
    st.info("Select two players to enable the AI comparison.")
