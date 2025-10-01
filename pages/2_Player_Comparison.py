import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from datetime import datetime
from auth import check_password
from branding import show_branding

# ---------- Protect page ----------
if not check_password():
    st.stop()

# ---------- Branding ----------
show_branding()
st.title("Player Comparison Page")

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
            "Passing%", "Pr. Pass% Dif.", "Successful Crosses", "Crossing%", "Deep Progressions",
            "Successful Dribbles", "Turnovers", "OBV", "Pass OBV",
            "Defensive Actions", "Aerial Win%", "PAdj Pressures",
            "PAdj Tack&Int", "Dribbles Stopped%", "Aggressive Actions", "Player Season Ball Recoveries 90"
        ],
        "groups": {
            "Passing%": "Possession",
            "Pr. Pass% Dif.": "Possession",
            "Successful Crosses": "Possession",
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
            cm_as_6 = cm_rows.copy(); cm_as_6["Six-Group Position"] = "Number 6"
            cm_as_8 = cm_rows.copy(); cm_as_8["Six-Group Position"] = "Number 8"
            df = pd.concat([df, cm_as_6, cm_as_8], ignore_index=True)

    return df

# ---------- Data path ----------
ROOT_DIR = Path(__file__).parent.parent
DATA_PATH = ROOT_DIR / "statsbomb_player_stats_clean.csv"

# ---------- Load + preprocess ----------
df_all_raw = load_statsbomb(DATA_PATH, _sig=_data_signature(DATA_PATH))
df_all_raw.columns = (
    df_all_raw.columns.astype(str)
    .str.strip()
    .str.replace(u"\xa0", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
)
df_all = preprocess_df(df_all_raw)
df = df_all.copy()

# ---------- League filter ----------
league_col = "Competition_norm" if "Competition_norm" in df.columns else "Competition"
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
    "Leagues",
    options=all_leagues,
    default=st.session_state.league_selection,
    key="league_selection",
    label_visibility="collapsed"
)

if selected_leagues:
    df = df[df[league_col].isin(selected_leagues)].copy()
    st.caption(f"Leagues selected: {len(selected_leagues)} | Players: {len(df)}")
    if df.empty:
        st.warning("No players match these leagues.")
        st.stop()
else:
    st.stop()

# ---------- Minutes + Age filters ----------
minutes_col = "Minutes played"
if minutes_col not in df.columns:
    df[minutes_col] = np.nan

c1, c2 = st.columns(2)
with c1:
    min_minutes = st.number_input("Minimum minutes", min_value=0, value=1000, step=50)
    df["_minutes_numeric"] = pd.to_numeric(df[minutes_col], errors="coerce")
    df = df[df["_minutes_numeric"] >= min_minutes].copy()
with c2:
    if "Age" in df.columns:
        df["_age_numeric"] = pd.to_numeric(df["Age"], errors="coerce")
        if df["_age_numeric"].notna().any():
            age_min = int(np.nanmin(df["_age_numeric"]))
            age_max = int(np.nanmax(df["_age_numeric"]))
            sel_min, sel_max = st.slider("Age range", min_value=age_min, max_value=age_max,
                                         value=(age_min, age_max), step=1)
            df = df[df["_age_numeric"].between(sel_min, sel_max)].copy()

st.caption(f"Filtering on '{minutes_col}' ≥ {min_minutes}. Players remaining {len(df)}")

# ---------- Position group ----------
st.markdown("#### 🟡 Select Position Group")
SIX_GROUPS = list(position_metrics.keys())
available_groups = [g for g in SIX_GROUPS if "Six-Group Position" in df.columns and g in df["Six-Group Position"].unique()]

selected_groups = st.multiselect(
    "Position Groups",
    options=available_groups,
    default=[],
    label_visibility="collapsed"
)

if selected_groups:
    df = df[df["Six-Group Position"].isin(selected_groups)].copy()
if df.empty:
    st.stop()

current_single_group = selected_groups[0] if len(selected_groups) == 1 else None

# ---------- Template ----------
st.markdown("#### 📊 Choose Radar Template")
template_names = list(position_metrics.keys())
if "template_select" not in st.session_state:
    st.session_state.template_select = template_names[0]

selected_position_template = st.selectbox(
    "Radar Template",
    template_names,
    index=template_names.index(st.session_state.template_select),
    key="template_select",
    label_visibility="collapsed"
)

metrics = position_metrics[selected_position_template]["metrics"]
metric_groups = position_metrics[selected_position_template]["groups"]

for m in metrics:
    if m not in df.columns:
        df[m] = 0
    df[m] = pd.to_numeric(df[m], errors="coerce").fillna(0)

# ---------- Player A & B ----------
players = df["Player"].dropna().unique().tolist()
if not players:
    st.stop()

if "cmpA" not in st.session_state:
    st.session_state.cmpA = players[0]
if "cmpB" not in st.session_state:
    st.session_state.cmpB = players[1] if len(players) > 1 else None

c1, c2 = st.columns(2)
with c1:
    pA = st.selectbox("Player A", players,
                      index=players.index(st.session_state.cmpA) if st.session_state.cmpA in players else 0,
                      key="cmpA_sel")
    st.session_state.cmpA = pA
with c2:
    pB = st.selectbox("Player B (optional)", ["(none)"] + players,
                      index=(players.index(st.session_state.cmpB) + 1) if st.session_state.cmpB in players else 0,
                      key="cmpB_sel")
    pB = None if pB == "(none)" else pB
    st.session_state.cmpB = pB

# ---------- Percentiles ----------
def compute_percentiles(metrics_list, group_df):
    bench = group_df.copy()
    for m in metrics_list:
        if m not in bench.columns:
            bench[m] = np.nan
        bench[m] = pd.to_numeric(bench[m], errors="coerce")
    raw = bench[metrics_list].copy()
    pct = (raw.rank(pct=True) * 100.0).round(1)
    return raw, pct

raw_df, pct_df = compute_percentiles(metrics, df)
rowA_pct = pct_df.loc[df["Player"] == pA, metrics].iloc[0] if pA in df["Player"].values else None
rowB_pct = pct_df.loc[df["Player"] == pB, metrics].iloc[0] if pB and pB in df["Player"].values else None

# ---------- Radar compare ----------
def radar_compare(labels, A_vals, B_vals=None, A_name="A", B_name="B",
                  labels_to_genre=None, genre_colors=None):
    import matplotlib.patches as mpatches

    if not labels:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No metrics", ha="center", va="center")
        return fig

    # Colours to match Livingston badge
    color_A = "#FFD700"  # Yellow
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

    # Title with A vs B, coloured to match lines
    if B_name:
        ax.set_title(f"{A_name} vs {B_name}",
                     fontsize=16, fontweight="bold", pad=30)

        # manually override colors by drawing text
        ax.text(0.45, 1.08, A_name,
                transform=ax.transAxes,
                ha="right", va="center",
                fontsize=16, fontweight="bold", color=color_A)

        ax.text(0.45, 1.08, f" vs {B_name}",
                transform=ax.transAxes,
                ha="left", va="center",
                fontsize=16, fontweight="bold", color=color_B)
    else:
        ax.set_title(A_name,
                     fontsize=16, fontweight="bold", color=color_A, pad=30)

    # Legend for metric groups at the bottom
    if labels_to_genre and genre_colors:
        present_groups = sorted(set(labels_to_genre.values()))
        patches = [mpatches.Patch(color=genre_colors[g], label=g)
                   for g in present_groups if g in genre_colors]
        ax.legend(handles=patches,
                  loc="upper center", bbox_to_anchor=(0.5, -0.08),
                  ncol=len(patches), frameon=False)

    return fig

labels_clean = [m.replace(" per 90", "").replace(", %", " (%)") for m in metrics]
labels_to_genre = {lbl: metric_groups[m] for lbl, m in zip(labels_clean, metrics)}

A_vals = rowA_pct.values if rowA_pct is not None else np.zeros(len(metrics))
B_vals = rowB_pct.values if rowB_pct is not None else None

fig = radar_compare(
    labels_clean, A_vals, B_vals,
    A_name=pA, B_name=pB,
    labels_to_genre=labels_to_genre,
    genre_colors=group_colors   # <-- FIXED
)
st.pyplot(fig, use_container_width=True)
