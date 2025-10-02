# data_loader.py

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re

# ---------- League name normalisation ----------
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
    "NB I": "Hungary NB I",
    "Besta deild karla": "Iceland Besta Deild",
    "Serie C": "Italy Serie C",
    "J2 League": "Japan J2 League",
    "Virsliga": "Latvia Virsliga",
    "A Lyga": "Lithuania A Lyga",
    "Botola Pro": "Morocco Botola Pro",
    "Eredivisie": "Eredivisie",
    "Eerste Divisie": "Netherlands Eerste Divisie",
    "1. Division": "Norway 1. Division",
    "Eliteserien": "Norway Eliteserien",
    "I Liga": "Poland 1 Liga",
    "Ekstraklasa": "Poland Ekstraklasa",
    "Segunda Liga": "Portugal Segunda Liga",
    "Liga Pro": "Portugal Segunda Liga",
    "Premier Division": "Republic of Ireland Premier Division",
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
    "Ligue 1": "Tunisia Ligue 1",
    "Ligue 1 (TUN)": "Tunisia Ligue 1",
    "Tunisia Ligue 1": "Tunisia Ligue 1",
    "France Ligue 1": "Tunisia Ligue 1",
    "USL Championship": "USA USL Championship",
    "Jupiler Pro League": "Jupiler Pro League",
    "Belgium Pro League": "Jupiler Pro League",
    "Belgian Pro League": "Jupiler Pro League",
    "Belgium Jupiler Pro League": "Jupiler Pro League",
}

# ---------- Position mapping ----------
def _clean_pos_token(tok: str) -> str:
    if pd.isna(tok):
        return ""
    t = str(tok).upper().strip()
    t = re.sub(r"[.\-_/]", " ", t)
    t = re.sub(r"\s+", "", t)
    return t

RAW_TO_SIX = {
    "RIGHTBACK": "Full Back", "LEFTBACK": "Full Back",
    "RIGHTWINGBACK": "Full Back", "LEFTWINGBACK": "Full Back",
    "RIGHTCENTREBACK": "Centre Back", "LEFTCENTREBACK": "Centre Back", "CENTREBACK": "Centre Back",
    "CENTREMIDFIELDER": "Centre Midfield",
    "RIGHTCENTREMIDFIELDER": "Centre Midfield", "LEFTCENTREMIDFIELDER": "Centre Midfield",
    "DEFENSIVEMIDFIELDER": "Number 6", "RIGHTDEFENSIVEMIDFIELDER": "Number 6", "LEFTDEFENSIVEMIDFIELDER": "Number 6",
    "CENTREATTACKINGMIDFIELDER": "Number 8", "ATTACKINGMIDFIELDER": "Number 8",
    "SECONDSTRIKER": "Number 8", "10": "Number 8",
    "RIGHTWING": "Winger", "LEFTWING": "Winger",
    "RIGHTMIDFIELDER": "Winger", "LEFTMIDFIELDER": "Winger",
    "CENTREFORWARD": "Striker", "RIGHTCENTREFORWARD": "Striker", "LEFTCENTREFORWARD": "Striker",
    "GOALKEEPER": "Goalkeeper",
}

def map_first_position_to_group(primary_pos_cell) -> str:
    tok = _clean_pos_token(primary_pos_cell)
    return RAW_TO_SIX.get(tok, None)

# ---------- Loader ----------
def load_one_file(p: Path) -> pd.DataFrame:
    if p.suffix.lower() in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(p, engine="openpyxl")
        except Exception:
            return pd.read_excel(p)  # fallback
    else:
        for kwargs in [dict(sep=None, engine="python"), {}, dict(encoding="latin1")]:
            try:
                return pd.read_csv(p, **kwargs)
            except Exception:
                continue
    raise ValueError(f"Unsupported or unreadable file: {p.name}")

def load_statsbomb(path: Path) -> pd.DataFrame:
    if path.is_file():
        df = load_one_file(path)
    else:
        files = sorted([f for f in path.iterdir() if f.is_file()])
        frames = [load_one_file(f) for f in files]
        df = pd.concat(frames, ignore_index=True, sort=False)
    return df

# ---------- Preprocessing ----------
def preprocess_df(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    # Clean headers
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(u"\xa0", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )

    # Competition normalisation
    if "Competition" in df.columns:
        df["Competition_norm"] = (
            df["Competition"].astype(str).str.strip().map(lambda x: LEAGUE_SYNONYMS.get(x, x))
        )
    else:
        df["Competition_norm"] = np.nan

    # Rename identifiers
    rename_map = {}
    if "Name" in df.columns: rename_map["Name"] = "Player"
    if "Primary Position" in df.columns: rename_map["Primary Position"] = "Position"
    if "Minutes" in df.columns: rename_map["Minutes"] = "Minutes played"
    df.rename(columns=rename_map, inplace=True)

    # Add age
    if "Birth Date" in df.columns:
        today = datetime.today()
        df["Age"] = pd.to_datetime(df["Birth Date"], errors="coerce").apply(
            lambda dob: today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            if pd.notna(dob) else np.nan
        )

    # Six-Group Position mapping
    if "Position" in df.columns:
        df["Six-Group Position"] = df["Position"].apply(map_first_position_to_group)
    else:
        df["Six-Group Position"] = np.nan

    # Duplicate generic CMs into both 6 & 8
    if "Six-Group Position" in df.columns:
        cm_mask = df["Six-Group Position"] == "Centre Midfield"
        if cm_mask.any():
            cm_rows = df.loc[cm_mask].copy()
            cm_as_6 = cm_rows.copy(); cm_as_6["Six-Group Position"] = "Number 6"
            cm_as_8 = cm_rows.copy(); cm_as_8["Six-Group Position"] = "Number 8"
            df = pd.concat([df, cm_as_6, cm_as_8], ignore_index=True)

    return df

# ---------- Combined loader + preprocess ----------
def load_and_preprocess(path: Path) -> pd.DataFrame:
    raw = load_statsbomb(path)
    return preprocess_df(raw)
