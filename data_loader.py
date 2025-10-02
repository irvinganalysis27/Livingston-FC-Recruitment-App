# data_loader.py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re

# ---------- League synonyms ----------
LEAGUE_SYNONYMS = {
    "A-League": "Australia A-League Men",
    "2. Liga": "Austria 2. Liga",
    "Challenger Pro League": "Belgium Challenger Pro League",
    "First League": "Bulgaria First League",
    "1. HNL": "Croatia 1. HNL",
    "HNL": "Croatia 1. HNL",
    "Czech Liga": "Czech First Tier",
    "Superliga": "Denmark Superliga",
    "League One": "England League One",
    "League Two": "England League Two",
    "National League": "England National League",
    "National League N / S": "England National League N/S",
    "Veikkausliiga": "Finland Veikkausliiga",
    "Championnat National": "France National 1",
    "Ligue 2": "Ligue 2",
    "2. Bundesliga": "2. Bundesliga",
    "3. Liga": "Germany 3. Liga",
    "Super League": "Greece Super League 1",
    "NB I": "Hungary NB I",
    "Besta deild karla": "Iceland Besta Deild",
    "Serie C": "Italy Serie C",
    "J2 League": "Japan J2 League",
    "Botola Pro": "Morocco Botola Pro",
    "Eredivisie": "Eredivisie",
    "Eerste Divisie": "Netherlands Eerste Divisie",
    "Eliteserien": "Norway Eliteserien",
    "I Liga": "Poland 1 Liga",
    "Ekstraklasa": "Poland Ekstraklasa",
    "Segunda Liga": "Portugal Segunda Liga",
    "Premier Division": "Republic of Ireland Premier Division",
    "Liga 1": "Romania Liga 1",
    "Championship": "Scotland Championship",
    "Premiership": "Scotland Premiership",
    "Super Liga": "Serbia Super Liga",
    "Slovakia First League": "Slovakia 1. Liga",
    "1. Liga (SVN)": "Slovenia 1. Liga",
    "PSL": "South Africa Premier Division",
    "Allsvenskan": "Sweden Allsvenskan",
    "Superettan": "Sweden Superettan",
    "Challenge League": "Switzerland Challenge League",
    "Ligue 1 (TUN)": "Tunisia Ligue 1",
    "USL Championship": "USA USL Championship",
    "Jupiler Pro League": "Jupiler Pro League",
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

# ---------- File loading ----------
def load_one_file(p: Path) -> pd.DataFrame:
    def try_excel():
        try:
            import openpyxl
            return pd.read_excel(p, engine="openpyxl")
        except Exception:
            return None
    def try_csv():
        for kwargs in [dict(sep=None, engine="python"), dict(), dict(encoding="latin1")]:
            try:
                return pd.read_csv(p, **kwargs)
            except Exception:
                continue
        return None

    if p.suffix.lower() in {".xlsx", ".xls"}:
        df = try_excel() or try_csv()
    else:
        df = try_csv() or try_excel()
    if df is None:
        raise ValueError(f"Unsupported or unreadable file: {p.name}")
    return df

def _data_signature(path: Path):
    path = Path(path)
    if path.is_file():
        s = path.stat()
        return ("file", str(path.resolve()), s.st_size, int(s.st_mtime))
    else:
        sigs = []
        for f in sorted(path.iterdir()):
            if f.is_file() and f.suffix.lower() in {".csv", ".xlsx", ".xls", ""}:
                try:
                    s = f.stat()
                    sigs.append((str(f.resolve()), s.st_size, int(s.st_mtime)))
                except FileNotFoundError:
                    continue
        return ("dir", str(path.resolve()), tuple(sigs))

def add_age_column(df: pd.DataFrame) -> pd.DataFrame:
    if "birth_date" not in df.columns:
        df["Age"] = np.nan
        return df
    today = datetime.today()
    df["Age"] = pd.to_datetime(df["birth_date"], errors="coerce").apply(
        lambda dob: today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        if pd.notna(dob) else np.nan
    )
    return df

def load_statsbomb(path: Path, _sig=None) -> pd.DataFrame:
    if path.is_file():
        df = load_one_file(path)
    else:
        frames = [load_one_file(f) for f in sorted(path.iterdir()) if f.is_file()]
        df = pd.concat(frames, ignore_index=True, sort=False)
    df = add_age_column(df)
    return df

def preprocess_df(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    # Normalise Competition
    if "Competition" in df.columns:
        df["Competition_norm"] = df["Competition"].astype(str).str.strip().map(
            lambda x: LEAGUE_SYNONYMS.get(x, x)
        )
    else:
        df["Competition_norm"] = np.nan

    # Rename basics
    rename_map = {}
    if "Name" in df.columns: rename_map["Name"] = "Player"
    if "Primary Position" in df.columns: rename_map["Primary Position"] = "Position"
    if "Minutes" in df.columns: rename_map["Minutes"] = "Minutes played"
    df.rename(columns=rename_map, inplace=True)

    # Six-group mapping
    if "Position" in df.columns:
        df["Six-Group Position"] = df["Position"].apply(map_first_position_to_group)
    else:
        df["Six-Group Position"] = np.nan

    # Duplicate CMs into 6 & 8
    if "Six-Group Position" in df.columns:
        cm_mask = df["Six-Group Position"] == "Centre Midfield"
        if cm_mask.any():
            cm_rows = df.loc[cm_mask].copy()
            cm_as_6 = cm_rows.copy(); cm_as_6["Six-Group Position"] = "Number 6"
            cm_as_8 = cm_rows.copy(); cm_as_8["Six-Group Position"] = "Number 8"
            df = pd.concat([df, cm_as_6, cm_as_8], ignore_index=True)

    return df
