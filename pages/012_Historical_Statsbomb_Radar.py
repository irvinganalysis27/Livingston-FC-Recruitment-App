# pages/Historical Leagues - statsbombs radar.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from PIL import Image
from datetime import datetime
from auth import check_password
from branding import show_branding
from ui.sidebar import render_sidebar

st.set_page_config(page_title="Historical Leagues - statsbombs radar", layout="centered")


# ---------- Auth ----------
if not check_password():
    st.stop()
render_sidebar()

# ---------- Branding ----------
show_branding()
st.title("Historical League Radars")

# ========== Helpers ==========
APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent
DATA_DIR = ROOT_DIR / "data" / "statsbomb"

SIX_GROUPS = ["Full Back", "Centre Back", "Number 6", "Number 8", "Number 10", "Winger", "Striker"]

def _clean_pos_token(tok: str) -> str:
    if pd.isna(tok): return ""
    t = str(tok).upper().strip()
    t = re.sub(r"[.\-_/]", " ", t)
    return re.sub(r"\s+", "", t)

RAW_TO_SIX = {
    "RIGHTBACK": "Full Back", "LEFTBACK": "Full Back",
    "RIGHTWINGBACK": "Full Back", "LEFTWINGBACK": "Full Back",
    "CENTREBACK": "Centre Back", "RIGHTCENTREBACK": "Centre Back", "LEFTCENTREBACK": "Centre Back",
    "DEFENSIVEMIDFIELDER": "Number 6", "RIGHTDEFENSIVEMIDFIELDER": "Number 6",
    "LEFTDEFENSIVEMIDFIELDER": "Number 6", "CENTREDEFENSIVEMIDFIELDER": "Number 6",
    "CENTREMIDFIELDER": "Centre Midfield", "RIGHTCENTREMIDFIELDER": "Centre Midfield", "LEFTCENTREMIDFIELDER": "Centre Midfield",
    "CENTREATTACKINGMIDFIELDER": "Number 8", "RIGHTATTACKINGMIDFIELDER": "Number 8", "LEFTATTACKINGMIDFIELDER": "Number 8",
    "LEFTWING": "Winger", "RIGHTWING": "Winger",
    "LEFTMIDFIELDER": "Winger", "RIGHTMIDFIELDER": "Winger",
    "CENTREFORWARD": "Striker", "LEFTCENTREFORWARD": "Striker", "RIGHTCENTREFORWARD": "Striker",
    "SECONDSTRIKER": "Striker", "10": "Striker",
}

def map_positions_to_groups(pos_cell):
    if pd.isna(pos_cell):
        return []
    tokens = [
        _clean_pos_token(t)
        for t in str(pos_cell).split(",")
        if _clean_pos_token(t)
    ]
    groups = []
    for tok in tokens:
        g = RAW_TO_SIX.get(tok)
        if g and g not in groups:
            groups.append(g)
    return groups

# Position templates (same as your StatsBomb radar)
position_metrics = {
    "Centre Back": {
        "metrics": [
            "NP Goals",
            "Passing%", "Pass OBV", "Pr. Long Balls", "UPr. Long Balls", "OBV", "Pr. Pass% Dif.",
            "PAdj Interceptions", "PAdj Tackles", "Dribbles Stopped%",
            "Defensive Actions", "Aggressive Actions", "Fouls",
            "Aerial Wins", "Aerial Win%"
        ],
        "groups": {
            "PAdj Interceptions": "Defensive", "PAdj Tackles": "Defensive", "Dribbles Stopped%": "Defensive",
            "Defensive Actions": "Defensive", "Aggressive Actions": "Defensive", "Fouls": "Defensive",
            "Aerial Wins": "Defensive", "Aerial Win%": "Defensive",
            "Passing%": "Possession", "Pr. Pass% Dif.": "Possession", "Pr. Long Balls": "Possession",
            "UPr. Long Balls": "Possession", "OBV": "Possession", "Pass OBV": "Possession",
            "NP Goals": "Attacking",
        }
    },
    "Full Back": {
        "metrics": [
            "Passing%", "Pr. Pass% Dif.", "Successful Box Cross%", "Deep Progressions",
            "Successful Dribbles", "Turnovers", "OBV", "Pass OBV",
            "Defensive Actions", "Aerial Win%", "PAdj Pressures", "PAdj Tack&Int",
            "Dribbles Stopped%", "Aggressive Actions", "Player Season Ball Recoveries 90"
        ],
        "groups": {
            "Passing%": "Possession", "Pr. Pass% Dif.": "Possession", "Successful Box Cross%": "Possession",
            "Deep Progressions": "Possession", "Successful Dribbles": "Possession", "Turnovers": "Possession",
            "OBV": "Possession", "Pass OBV": "Possession",
            "Defensive Actions": "Defensive", "Aerial Win%": "Defensive",
            "PAdj Pressures": "Defensive", "PAdj Tack&Int": "Defensive",
            "Dribbles Stopped%": "Defensive", "Aggressive Actions": "Defensive",
            "Player Season Ball Recoveries 90": "Defensive",
        }
    },
    "Number 6": {
        "metrics": [
            "xGBuildup", "xG Assisted",
            "Passing%", "Deep Progressions", "Turnovers", "OBV", "Pass OBV", "Pr. Pass% Dif.",
            "PAdj Interceptions", "PAdj Tackles", "Dribbles Stopped%", "Aggressive Actions",
            "Aerial Win%", "Player Season Ball Recoveries 90", "Pressure Regains",
        ],
        "groups": {
            "xGBuildup": "Attacking", "xG Assisted": "Attacking",
            "Passing%": "Possession", "Deep Progressions": "Possession", "Turnovers": "Possession",
            "OBV": "Possession", "Pass OBV": "Possession", "Pr. Pass% Dif.": "Possession",
            "PAdj Interceptions": "Defensive", "PAdj Tackles": "Defensive", "Dribbles Stopped%": "Defensive",
            "Aggressive Actions": "Defensive", "Aerial Win%": "Defensive", "Player Season Ball Recoveries 90": "Defensive",
            "Pressure Regains": "Defensive",
        }
    },
    "Number 8": {
        "metrics": [
            "xGBuildup", "xG Assisted", "Shots", "xG", "NP Goals",
            "Passing%", "Deep Progressions", "OP Passes Into Box", "Pass OBV", "OBV", "Deep Completions",
            "Pressure Regains", "PAdj Pressures", "Player Season Fhalf Ball Recoveries 90",
            "Aggressive Actions",
        ],
        "groups": {
            "xGBuildup": "Attacking", "xG Assisted": "Attacking", "Shots": "Attacking", "xG": "Attacking", "NP Goals": "Attacking",
            "Passing%": "Possession", "Deep Progressions": "Possession", "OP Passes Into Box": "Possession",
            "Pass OBV": "Possession", "OBV": "Possession", "Deep Completions": "Possession",
            "Pressure Regains": "Defensive", "PAdj Pressures": "Defensive",
            "Player Season Fhalf Ball Recoveries 90": "Defensive", "Aggressive Actions": "Defensive",
        }
    },
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
            "OBV": "Possession",
            "OP Passes Into Box": "Possession",
            "OP Key Passes": "Possession",
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
    "Winger": {
        "metrics": [
            "xG", "Shots", "xG/Shot", "Touches In Box", "OP xG Assisted", "NP Goals",
            "OP Passes Into Box", "Successful Box Cross%", "Passing%",
            "Successful Dribbles", "Turnovers", "OBV", "D&C OBV", "Fouls Won", "Deep Progressions",
            "Player Season Fhalf Pressures 90",
        ],
        "groups": {
            "NP Goals": "Attacking", "xG": "Attacking", "Shots": "Attacking", "xG/Shot": "Attacking",
            "Touches In Box": "Attacking", "OP xG Assisted": "Attacking",
            "OP Passes Into Box": "Possession", "Successful Box Cross%": "Possession", "Passing%": "Possession",
            "Successful Dribbles": "Possession", "Fouls Won": "Possession", "Turnovers": "Possession",
            "OBV": "Possession", "D&C OBV": "Possession", "Deep Progressions": "Possession",
            "Player Season Fhalf Pressures 90": "Defensive",
        }
    },
    "Striker": {
        "metrics": [
            "Aggressive Actions", "NP Goals", "xG", "Shots", "xG/Shot",
            "Goal Conversion%", "Touches In Box", "xG Assisted",
            "Fouls Won", "Deep Completions", "OP Key Passes",
            "Aerial Win%", "Aerial Wins", "Player Season Fhalf Pressures 90",
        ],
        "groups": {
            "NP Goals": "Attacking", "xG": "Attacking", "Shots": "Attacking", "xG/Shot": "Attacking",
            "Goal Conversion%": "Attacking", "Touches In Box": "Attacking", "xG Assisted": "Attacking",
            "Fouls Won": "Possession", "Deep Completions": "Possession", "OP Key Passes": "Possession",
            "Aerial Win%": "Defensive", "Aerial Wins": "Defensive", "Aggressive Actions": "Defensive",
            "Player Season Fhalf Pressures 90": "Defensive",
        }
    },
}

LOWER_IS_BETTER = {"Turnovers", "Fouls", "Pr. Long Balls", "UPr. Long Balls"}

# --- Percentile calculation (handles inverted metrics) ---
def pct_rank(series: pd.Series, lower_is_better: bool) -> pd.Series:
    """
    Convert a numeric series into 0–100 percentiles.
    If lower_is_better=True, lower raw values get higher percentile ranks.
    """
    series = pd.to_numeric(series, errors="coerce").fillna(0)
    ranks = series.rank(pct=True, ascending=True)
    if lower_is_better:
        percentiles = 1.0 - ranks  # invert if smaller = better
    else:
        percentiles = ranks
    return (percentiles * 100).round(1)

def open_image(path: Path):
    try:
        return Image.open(path)
    except Exception:
        return None

def prettify_league_name(slug: str) -> str:
    name = slug.replace("_", " ").title()

    # Common football-specific fixes
    replacements = {
        "Usa": "USA",
        "Usl": "USL",
        "A League": "A-League",
        "1 Liga": "1. Liga",
        "2 Liga": "2. Liga",
        "3 Liga": "3. Liga",
        "Liga 1": "Liga 1",
        "Liga 3": "Liga 3",
        "Sub 23": "Sub-23",
        "Premier Division": "Premier Division",
        "First League": "First League",
        "First Tier": "First Tier",
        "Super Liga": "Super Liga",
        "Championship": "Championship",
        "Ekstraklasa": "Ekstraklasa",
        "Challenge League": "Challenge League",
    }

    for k, v in replacements.items():
        name = name.replace(k, v)

    return name

# ========== League & Season Selection (disk-based) ==========
st.markdown("### League & Season Selection")

league_dirs = sorted([
    p.name for p in DATA_DIR.iterdir()
    if p.is_dir()
])

if not league_dirs:
    st.error("No leagues found in data/statsbomb.")
    st.stop()

league_display_map = {p: prettify_league_name(p) for p in league_dirs}
display_to_slug = {v: k for k, v in league_display_map.items()}

selected_display_league = st.selectbox(
    "League",
    sorted(league_display_map.values()),
    key="historical_league_select"
)

selected_league = display_to_slug[selected_display_league]

league_path = DATA_DIR / selected_league

season_files = sorted([
    f.name for f in league_path.glob("*_clean.csv")
])

if not season_files:
    st.error(f"No cleaned season files found for {selected_league}.")
    st.stop()

season_labels = [f.replace("_clean.csv", "") for f in season_files]

selected_season = st.selectbox(
    "Season",
    season_labels,
    key="historical_season_select"
)

@st.cache_data(show_spinner=False)
def load_season_data(league: str, season: str) -> pd.DataFrame:
    path = DATA_DIR / league / f"{season}_clean.csv"
    return pd.read_csv(path)

df = load_season_data(selected_league, selected_season)

if df.empty:
    st.error("Selected season file is empty.")
    st.stop()

# ========== Light preprocessing (names, positions, etc.) ==========
df.columns = (
    df.columns.astype(str)
      .str.strip()
      .str.replace(u"\xa0", " ", regex=False)
      .str.replace(r"\s+", " ", regex=True)
)

# ============================================================
# 🪪 3. Rename Identifiers & NORMALISE STATSBOMB METRICS
# ============================================================
rename_map = {}

# --- Core identifiers ---
if "Name" in df.columns:
    rename_map["Name"] = "Player"
if "Primary Position" in df.columns:
    rename_map["Primary Position"] = "Position"
if "Minutes" in df.columns:
    rename_map["Minutes"] = "Minutes played"

# --- Passing & pressure behaviour ---
rename_map.update({
    "Successful Box Cross %": "Successful Box Cross%",
    "Player Season Box Cross Ratio": "Successful Box Cross%",
    "Player Season Change In Passing Ratio": "Pr. Pass% Dif.",
    "Player Season Pressured Long Balls 90": "Pr. Long Balls",
    "Player Season Unpressured Long Balls 90": "UPr. Long Balls",
})

# --- xG buildup (CRITICAL for 6s / 8s historical data) ---
rename_map.update({
    "Player Season Xgbuildup 90": "xGBuildup",
    "Player Season Op Xgbuildup 90": "xGBuildup",
    "Player Season Xgbuildup": "xGBuildup",
    "Player Season Op Xgbuildup": "xGBuildup",
})

# --- Defensive / pressure location ---
rename_map.update({
    "Player Season F3 Pressures 90": "Pressures in Final 1/3",
    "Player Season Fhalf Pressures 90": "Player Season Fhalf Pressures 90",
    "Player Season Fhalf Ball Recoveries 90": "Player Season Fhalf Ball Recoveries 90",
    "Player Season Ball Recoveries 90": "Player Season Ball Recoveries 90",
})

# --- Possession progression naming consistency ---
rename_map.update({
    "Progressions to Final 1/3": "Deep Progressions",
    "Completed Passes Final 1/3": "Deep Completions",
    "Lost Balls": "Turnovers",
})

df.rename(columns=rename_map, inplace=True)

# ✅ Guard against duplicate column names created by normalisation (common in historical exports)
# Example: multiple xGBuildup source columns can become duplicate `xGBuildup` after rename.
if df.columns.duplicated().any():
    df = df.loc[:, ~df.columns.duplicated()].copy()


if "Primary Position" in df.columns:
    if "Secondary Position" in df.columns:
        df["Positions played"] = df["Primary Position"].fillna("") + np.where(
            df["Secondary Position"].notna() & (df["Secondary Position"] != ""),
            ", " + df["Secondary Position"].astype(str),
            ""
        )
    else:
        df["Positions played"] = df["Primary Position"]
elif "Position" in df.columns:
    df["Positions played"] = df["Position"]
else:
    df["Positions played"] = np.nan

# --- Derived / calculated metrics ---
if "Player Season Total Dribbles 90" in df.columns and "Player Season Failed Dribbles 90" in df.columns:
    df["Successful Dribbles"] = (
        pd.to_numeric(df["Player Season Total Dribbles 90"], errors="coerce").fillna(0)
        - pd.to_numeric(df["Player Season Failed Dribbles 90"], errors="coerce").fillna(0)
    )

if "Successful Box Cross %" in df.columns:
    df.rename(columns={"Successful Box Cross %": "Successful Box Cross%"}, inplace=True)
if "Player Season Box Cross Ratio" in df.columns:
    df.rename(columns={"Player Season Box Cross Ratio": "Successful Box Cross%"}, inplace=True)



# Age derivation if Birth Date exists
if "Birth Date" in df.columns and "Age" not in df.columns:
    today = datetime.today()
    dob = pd.to_datetime(df["Birth Date"], errors="coerce")
    df["Age"] = dob.apply(
        lambda d: today.year - d.year - ((today.month, today.day) < (d.month, d.day))
        if pd.notnull(d)
        else np.nan
    )

# ==================== DEDUPE PLAYERS BY MOST RECENT MATCH (Option B) ====================
match_col_candidates = [
    "Player Season Most Recent Match",
    "player_season_most_recent_match",
    "Most Recent Match"
]
match_col = next((c for c in match_col_candidates if c in df.columns), None)

if match_col:
    df["last_match_dt"] = pd.to_datetime(df[match_col], errors="coerce")
else:
    df["last_match_dt"] = pd.NaT

if "Player Id" in df.columns:
    df = df.sort_values(
        ["Player Id", "last_match_dt"],
        ascending=[True, False]
    )
    # At this point Six-Group Position may not exist yet, so dedupe only by Player Id
    df = df.drop_duplicates(
        subset=["Player Id"],
        keep="first"
    )

# Standard id columns
rename_map = {}
if "Name" in df.columns: rename_map["Name"] = "Player"
if "Primary Position" in df.columns: rename_map["Primary Position"] = "Position"
if "Minutes" in df.columns: rename_map["Minutes"] = "Minutes played"
df.rename(columns=rename_map, inplace=True)

# ✅ Guard against duplicate column names created by normalisation (common in historical exports)
# Example: multiple xGBuildup source columns can become duplicate `xGBuildup` after rename.
if df.columns.duplicated().any():
    df = df.loc[:, ~df.columns.duplicated()].copy()



# League column normalisation (no mapping needed)
if "Competition" not in df.columns:
    for alt in ["competition", "competition_name", "league", "league_name"]:
        if alt in df.columns:
            df.rename(columns={alt: "Competition"}, inplace=True)
            break

df["Competition_norm"] = df["Competition"].astype(str).str.strip() if "Competition" in df.columns else np.nan

# ================== POSITION → SIX-GROUP EXPANSION (AUTHORITATIVE) ==================

df["Six-Group Position"] = df.get("Positions played", "").apply(map_positions_to_groups)
df = df.explode("Six-Group Position").reset_index(drop=True)

# Final dedupe: player + role + season context
if "Player Id" in df.columns:
    df = df.drop_duplicates(
        subset=["Player Id", "Six-Group Position"],
        keep="first"
    )

# Age derivation if Birth Date exists
if "Birth Date" in df.columns and "Age" not in df.columns:
    today = datetime.today()
    dob = pd.to_datetime(df["Birth Date"], errors="coerce")
    df["Age"] = dob.apply(lambda d: today.year - d.year - ((today.month, today.day) < (d.month, d.day)) if pd.notnull(d) else np.nan)

# ========== Minutes + Age Filters (same layout as main radar) ==========

minutes_col = "Minutes played"
if minutes_col not in df.columns:
    df[minutes_col] = np.nan

df["_minutes_numeric"] = pd.to_numeric(df[minutes_col], errors="coerce")

# Determine dataset-wide max for minutes
max_minutes_available = int(np.nanmax(df["_minutes_numeric"])) if df["_minutes_numeric"].notna().any() else 0

# Initialise session state defaults
if "min_minutes_typed" not in st.session_state:
    st.session_state.min_minutes_typed = 500

if "max_minutes_typed" not in st.session_state:
    st.session_state.max_minutes_typed = max_minutes_available

st.markdown("### Minutes and Age")
c_min, c_max, c_age = st.columns([1.5, 1.5, 2])

with c_min:
    st.markdown("**From**")
    st.session_state.min_minutes_typed = st.number_input(
        " ",
        min_value=0,
        value=st.session_state.min_minutes_typed,
        step=50,
        key="min_minutes_input_typed",
        label_visibility="collapsed"
    )

with c_max:
    st.markdown("**To**")
    st.session_state.max_minutes_typed = st.number_input(
        " ",
        min_value=0,
        value=st.session_state.max_minutes_typed,
        step=50,
        key="max_minutes_input_typed",
        label_visibility="collapsed"
    )

# Apply minutes filter
df = df[
    (df["_minutes_numeric"] >= st.session_state.min_minutes_typed)
    & (df["_minutes_numeric"] <= st.session_state.max_minutes_typed)
].copy()

players_remaining_after_minutes = len(df)

with c_age:
    st.markdown("**Age**")
    if "Age" in df.columns:
        df["_age_numeric"] = pd.to_numeric(df["Age"], errors="coerce")
        if df["_age_numeric"].notna().any():
            age_min = int(np.nanmin(df["_age_numeric"]))
            age_max = int(np.nanmax(df["_age_numeric"]))

            if "age_range" not in st.session_state:
                st.session_state.age_range = (age_min, age_max)

            st.session_state.age_range = st.slider(
                " ",
                min_value=age_min,
                max_value=age_max,
                value=st.session_state.age_range,
                step=1,
                label_visibility="collapsed",
                key="historical_age_slider"
            )

            lo, hi = st.session_state.age_range
            df = df[df["_age_numeric"].between(lo, hi)].copy()

players_remaining = len(df)

st.markdown(f"**Players remaining: {players_remaining}**")

# Position groups
available_groups = [g for g in SIX_GROUPS if g in df["Six-Group Position"].dropna().unique().tolist()]
if "selected_groups" not in st.session_state:
    st.session_state.selected_groups = []
st.markdown("#### Select Position Group")
selected_groups = st.multiselect("Position Groups", options=available_groups, default=st.session_state.selected_groups, label_visibility="collapsed", key="pos_group_multiselect")
if selected_groups:
    df = df[df["Six-Group Position"].isin(selected_groups)].copy()
    if df.empty:
        st.warning("No players after group filter.")
        st.stop()

# ---------- Template chooser ----------
st.markdown("#### 📊 Choose Radar Template")

# Set up persistent state
if "template_select" not in st.session_state:
    st.session_state.template_select = list(position_metrics.keys())[0]
if "last_template_choice" not in st.session_state:
    st.session_state.last_template_choice = st.session_state.template_select
if "manual_override" not in st.session_state:
    st.session_state.manual_override = False
if "last_groups_tuple" not in st.session_state:
    st.session_state.last_groups_tuple = tuple()

# Auto-sync when group selection changes — but only if user hasn't manually changed template
if tuple(selected_groups) != st.session_state.last_groups_tuple:
    if len(selected_groups) == 1:
        pos = selected_groups[0]
        if pos in position_metrics:
            st.session_state.template_select = pos
            st.session_state.manual_override = False
    st.session_state.last_groups_tuple = tuple(selected_groups)

# Build list of template names
template_names = list(position_metrics.keys())
if st.session_state.template_select not in template_names:
    st.session_state.template_select = template_names[0]

# Template selector widget (lets user override manually)
selected_position_template = st.selectbox(
    "Radar Template",
    template_names,
    index=template_names.index(st.session_state.template_select),
    key="template_select",
    label_visibility="collapsed"
)

# Detect manual override (user changed the dropdown)
if st.session_state.template_select != st.session_state.last_template_choice:
    st.session_state.manual_override = True
    st.session_state.last_template_choice = st.session_state.template_select

# Ensure metric columns exist and numeric
metrics = position_metrics[selected_position_template]["metrics"]
metric_groups = position_metrics[selected_position_template]["groups"]
for m in metrics:
    if m not in df.columns:
        df[m] = 0
df[metrics] = df[metrics].apply(pd.to_numeric, errors="coerce").fillna(0)

# Essential Criteria (same UX, simple)
with st.expander("Essential Criteria", expanded=False):
    use_all_cols = st.checkbox("Pick from all numeric columns", value=False)
    numeric_cols_all = sorted([c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])
    metric_pool_base = numeric_cols_all if use_all_cols else metrics

    if "ec_rows" not in st.session_state:
        st.session_state.ec_rows = 1

    cbtn1, cbtn2, cbtn3 = st.columns(3)
    with cbtn1:
        if st.button("Add criterion"): st.session_state.ec_rows += 1
    with cbtn2:
        if st.button("Remove last", disabled=st.session_state.ec_rows <= 1):
            st.session_state.ec_rows = max(1, st.session_state.ec_rows - 1)
    with cbtn3:
        apply_all = st.checkbox("Apply all criteria", value=False)

    crit = []
    for i in range(st.session_state.ec_rows):
        st.markdown(f"**Criterion {i+1}**")
        c1, c2, c3, c4 = st.columns([3,2,2,3])
        with c1:
            metric_name = st.selectbox("Metric", metric_pool_base, key=f"ec_metric_{i}")
        with c2:
            mode = st.radio("Apply to", ["Raw", "Percentile"], horizontal=True, key=f"ec_mode_{i}")
        with c3:
            op = st.selectbox("Operator", [">=", ">", "<=", "<"], key=f"ec_op_{i}")
        with c4:
            if mode == "Percentile":
                default_thr = 50.0
            else:
                default_thr = float(np.nanmedian(pd.to_numeric(df[metric_name], errors="coerce")))
                if not np.isfinite(default_thr): default_thr = 0.0
            thr = st.text_input("Threshold", value=str(int(default_thr)), key=f"ec_thr_{i}")
            try:
                thr_val = float(thr)
            except ValueError:
                thr_val = default_thr
        crit.append((metric_name, mode, op, thr_val))

    if apply_all and crit:
        mask_all = pd.Series(True, index=df.index)
        temp_cols = []
        for metric_name, mode, op, thr_val in crit:
            if mode == "Percentile":
                perc_series = (pd.to_numeric(df[metric_name], errors="coerce").rank(pct=True) * 100).round(1)
                tmp = f"__pct__{metric_name}"
                df[tmp] = perc_series; temp_cols.append(tmp)
                col = tmp
            else:
                col = metric_name
            series = pd.to_numeric(df[col], errors="coerce")
            if op == ">=": mask = series >= thr_val
            elif op == ">": mask = series > thr_val
            elif op == "<=": mask = series <= thr_val
            else: mask = series < thr_val
            mask_all &= mask
        kept = int(mask_all.sum()); dropped = int((~mask_all).sum())
        df = df[mask_all].copy()
        if temp_cols: df.drop(columns=temp_cols, inplace=True, errors="ignore")
        st.caption(f"Applied criteria - kept {kept}, removed {dropped}.")

# Build plot_data and Wyscout-style Z
keep_cols = ["Player", "Team", "Age", "Height", "Positions played", "Minutes played", "Six-Group Position"]
for c in keep_cols:
    if c not in df.columns: df[c] = np.nan

metrics_df = df[metrics].copy()

# --- Calculate percentiles (respecting LOWER_IS_BETTER metrics) ---
percentiles = pd.DataFrame(index=metrics_df.index, columns=metrics_df.columns, dtype=float)
for m in metrics_df.columns:
    percentiles[m] = pct_rank(metrics_df[m], lower_is_better=(m in LOWER_IS_BETTER))

plot_data = pd.concat([df[keep_cols], metrics_df, percentiles.add_suffix(" (percentile)")], axis=1)

# Wyscout-style Avg Z from percentiles (center 50, sd 15)
z_from_pct = (percentiles - 50.0) / 15.0
plot_data["Avg Z Score"] = z_from_pct.mean(axis=1).fillna(0.0)

# Rank on Avg Z only (simple Wyscout logic)
plot_data.sort_values("Avg Z Score", ascending=False, inplace=True, ignore_index=True)
plot_data["Rank"] = np.arange(1, len(plot_data) + 1)

# Player selector
players = plot_data["Player"].dropna().unique().tolist()
if not players:
    st.warning("No players available after filters.")
    st.stop()

if "selected_player" not in st.session_state or st.session_state.selected_player not in players:
    st.session_state.selected_player = players[0]
selected_player = st.selectbox("Choose a player", players, index=players.index(st.session_state.selected_player), key="player_select")
st.session_state.selected_player = selected_player

# --- Display name overrides (match main radar) ---
DISPLAY_NAMES = {
    "Player Season Fhalf Pressures 90": "Pressures in Opposition Half",
    "Deep Completions": "Completed Passes Final 1/3",
    "Turnovers": "Lost Balls",
    "Deep Progressions": "Progressions to Final 1/3",
    "Player Season Fhalf Ball Recoveries 90": "Ball Recovery Opp. Half",
    "Player Season Ball Recoveries 90": "Ball Recoveries",
}

# ================== Radar Chart Display ==================
def plot_radial_bar_grouped(player_name, plot_data, metric_groups, group_colors=None):
    import matplotlib.patches as mpatches
    from matplotlib import colormaps as mcm
    import matplotlib.colors as mcolors

    if not isinstance(group_colors, dict) or len(group_colors) == 0:
        group_colors = {
            "Attacking": "crimson",
            "Possession": "seagreen",
            "Defensive": "royalblue",
            "Off The Ball": "dimgray",
            "Goalkeeping": "dimgray",
        }

    # --- Player row ---
    row_df = plot_data.loc[plot_data["Player"] == player_name]
    if row_df.empty:
        st.warning(f"No player named '{player_name}' found.")
        return
    row = row_df.iloc[0]

    # --- Metric order ---
    group_order = ["Possession", "Defensive", "Attacking", "Off The Ball", "Goalkeeping"]
    ordered_metrics = [m for g in group_order for m, gg in metric_groups.items() if gg == g]
    valid_metrics, valid_pcts = [], []
    for m in ordered_metrics:
        pct_col = f"{m} (percentile)"
        if m in row.index and pct_col in row.index:
            valid_metrics.append(m)
            valid_pcts.append(pct_col)
    if not valid_metrics:
        st.info("No valid metrics to plot for this player.")
        return

    # --- Numeric values ---
    raw_vals = pd.to_numeric(row[valid_metrics], errors="coerce").fillna(0).to_numpy()
    pct_vals = pd.to_numeric(row[valid_pcts], errors="coerce").fillna(50).to_numpy()

    n = len(valid_metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cmap = mcm.get_cmap("RdYlGn")
    norm = mcolors.Normalize(vmin=0, vmax=100)
    bar_colors = [cmap(norm(v)) for v in pct_vals]

    # --- Setup plot ---
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_ylim(0, 100); ax.set_yticklabels([]); ax.set_xticks([])
    ax.spines["polar"].set_visible(False)

    # --- Bars ---
    ax.bar(angles, pct_vals, width=2 * np.pi / n * 0.85, color=bar_colors,
           edgecolor="black", linewidth=0.6, alpha=0.9)

    # --- Raw values inside ---
    for ang, raw in zip(angles, raw_vals):
        txt = f"{raw:.2f}" if np.isfinite(raw) else "-"
        ax.text(ang, 50, txt, ha="center", va="center", fontsize=10, color="black", fontweight="bold")

    # --- Metric labels (StatsBomb-style outward horizontal) ---
    label_radius = 108
    for ang, m in zip(angles, valid_metrics):
        label = DISPLAY_NAMES.get(m, m)
        label = label.replace(" per 90", "").replace(", %", " (%)")
        color = group_colors.get(metric_groups.get(m, "Unknown"), "black")

        # Determine X,Y coordinates for label position
        x = label_radius * np.cos(ang - np.pi/2)
        y = label_radius * np.sin(ang - np.pi/2)

        # Center text horizontally
        ha = "center"
        va = "center"

        ax.text(
            np.deg2rad(np.degrees(ang)),  # same polar coordinate
            label_radius,
            label,
            color=color,
            fontsize=10,
            fontweight="bold",
            ha=ha,
            va=va,
            rotation=0,
            rotation_mode="anchor",
            transform=ax.transData
        )

    # --- Legend ---
    present_groups = list(dict.fromkeys([metric_groups.get(m, "Unknown") for m in valid_metrics]))
    patches = [mpatches.Patch(color=group_colors.get(g, "grey"), label=g) for g in present_groups]
    if patches:
        fig.subplots_adjust(top=0.86, bottom=0.08)
        ax.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.06),
                  ncol=min(len(patches), 4), frameon=False)

    # --- Title (player info identical style) ---
    avg_z = float(row.get("Avg Z Score", 0) or 0)
    age = row.get("Age", np.nan)
    height = row.get("Height", np.nan)
    team = row.get("Team within selected timeframe", "") or row.get("Team", "") or ""
    mins = row.get("Minutes played", np.nan)
    role = row.get("Six-Group Position", "") or ""
    comp = row.get("Competition_norm") or row.get("Competition") or ""
    rank_v = int(row.get("Rank", 0)) if pd.notnull(row.get("Rank", 0)) else None

    top_parts = [player_name]
    if role: top_parts.append(role)
    if pd.notnull(age): top_parts.append(f"{int(age)} years old")
    if pd.notnull(height): top_parts.append(f"{int(height)} cm")
    line1 = " | ".join(map(str, [p for p in top_parts if p]))

    bottom_parts = []
    if team: bottom_parts.append(team)
    if comp: bottom_parts.append(comp)
    if pd.notnull(mins): bottom_parts.append(f"{int(mins)} mins")
    if rank_v: bottom_parts.append(f"Rank #{rank_v}")
    bottom_parts.append(f"Z {avg_z:.2f}")

    # ensure all string-safe and filter empties
    bottom_parts = [str(p) for p in bottom_parts if p and str(p).strip() != ""]
    line2 = " | ".join(bottom_parts)

    ax.set_title(f"{line1}\n{line2}", color="black", size=22, pad=20, y=1.10)
    st.pyplot(fig, width="stretch")

# --- Draw chart ---
group_colors = {
    "Attacking": "crimson",
    "Possession": "seagreen",
    "Defensive": "royalblue",
    "Off The Ball": "dimgray",
    "Goalkeeping": "dimgray",
}
if st.session_state.selected_player:
    plot_radial_bar_grouped(st.session_state.selected_player, plot_data, metric_groups, group_colors)

# ================== Ranking Table (identical layout) ==================
st.markdown("### Players Ranked by Z-Score")

cols_for_table = [
    "Player", "Positions played", "Age", "Team",
    "Minutes played",
    "Avg Z Score", "Rank"
]

for c in cols_for_table:
    if c not in plot_data.columns:
        plot_data[c] = np.nan

ranking_df = (plot_data[cols_for_table]
              .sort_values(by="Avg Z Score", ascending=False)
              .reset_index(drop=True))
ranking_df.index = np.arange(1, len(ranking_df) + 1)
ranking_df.index.name = "Row"

# Clean + formatting
if "Age" in ranking_df:
    ranking_df["Age"] = ranking_df["Age"].apply(lambda x: int(x) if pd.notnull(x) else x)

# --- Display identical to StatsBomb table (non-editable version) ---
st.dataframe(ranking_df, use_container_width=True)
