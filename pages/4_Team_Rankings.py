# pages/3_Team_Rankings.py

import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime
import sqlite3

from auth import check_password
from branding import show_branding

# ============================================================
# Setup & Protection
# ============================================================
if not check_password():
    st.stop()

show_branding()
st.title("🏆 Team Player Rankings")

APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent
DATA_PATH = ROOT_DIR / "statsbomb_player_stats_clean.csv"
DB_PATH = APP_DIR / "favourites.db"

# ============================================================
# Favourites DB
# ============================================================
def _ensure_db():
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
_ensure_db()

def get_favourites():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT player, team, league, position FROM favourites").fetchall()
    conn.close()
    return rows

def add_favourite(player, team=None, league=None, position=None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT OR REPLACE INTO favourites (player, team, league, position) VALUES (?,?,?,?)",
                 (player, team, league, position))
    conn.commit()
    conn.close()

def remove_favourite(player):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM favourites WHERE player=?", (player,))
    conn.commit()
    conn.close()

# ============================================================
# Data Loading
# ============================================================
def load_one_file(p: Path) -> pd.DataFrame:
    return pd.read_excel(p) if p.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(p)

def load_statsbomb(path: Path) -> pd.DataFrame:
    if path.is_file():
        return load_one_file(path)
    frames = [load_one_file(f) for f in sorted(p for p in path.iterdir() if p.is_file())]
    return pd.concat(frames, ignore_index=True, sort=False)

def add_age_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Birth Date" in df.columns:
        today = datetime.today()
        df["Age"] = pd.to_datetime(df["Birth Date"], errors="coerce").apply(
            lambda dob: today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            if pd.notna(dob) else np.nan
        )
    return df

# ============================================================
# Position & League Mapping
# ============================================================
RAW_TO_GROUP = {
    "LEFTBACK": "Full Back", "LEFTWINGBACK": "Full Back",
    "RIGHTBACK": "Full Back", "RIGHTWINGBACK": "Full Back",
    "CENTREBACK": "Centre Back", "LEFTCENTREBACK": "Centre Back", "RIGHTCENTREBACK": "Centre Back",
    "DEFENSIVEMIDFIELDER": "Number 6", "LEFTDEFENSIVEMIDFIELDER": "Number 6",
    "RIGHTDEFENSIVEMIDFIELDER": "Number 6", "CENTREDEFENSIVEMIDFIELDER": "Number 6",
    "CENTREMIDFIELDER": "Number 8", "LEFTCENTREMIDFIELDER": "Number 8", "RIGHTCENTREMIDFIELDER": "Number 8",
    "CENTREATTACKINGMIDFIELDER": "Number 8", "RIGHTATTACKINGMIDFIELDER": "Number 8",
    "LEFTATTACKINGMIDFIELDER": "Number 8", "SECONDSTRIKER": "Number 8",
    "LEFTWING": "Winger", "LEFTMIDFIELDER": "Winger",
    "RIGHTWING": "Winger", "RIGHTMIDFIELDER": "Winger",
    "CENTREFORWARD": "Striker", "LEFTCENTREFORWARD": "Striker", "RIGHTCENTREFORWARD": "Striker",
    "GOALKEEPER": "Goalkeeper",
}
def _clean_pos_token(tok: str) -> str:
    if pd.isna(tok): return ""
    t = str(tok).upper().strip()
    t = re.sub(r"[.\-_/]", " ", t)
    return re.sub(r"\s+", "", t)

def map_first_position_to_group(primary_pos_cell) -> str:
    return RAW_TO_GROUP.get(_clean_pos_token(primary_pos_cell), None)

LOWER_IS_BETTER = {"Turnovers", "Fouls", "Pr. Long Balls", "UPr. Long Balls"}

# ============================================================
# Preprocessing (League + Multiplier + Position)
# ============================================================
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {}
    if "Name" in df.columns: rename_map["Name"] = "Player"
    if "Primary Position" in df.columns: rename_map["Primary Position"] = "Position"
    if "Minutes" in df.columns: rename_map["Minutes"] = "Minutes played"
    if rename_map: df.rename(columns=rename_map, inplace=True)

    if "Competition_norm" not in df.columns and "Competition" in df.columns:
        df["Competition_norm"] = df["Competition"].astype(str)

    # merge multipliers
    try:
        mult = pd.read_excel(ROOT_DIR / "league_multipliers.xlsx")
        if {"League", "Multiplier"}.issubset(mult.columns):
            df = df.merge(mult, left_on="Competition_norm", right_on="League", how="left")
            df["Multiplier"] = pd.to_numeric(df["Multiplier"], errors="coerce").fillna(1.0)
        else:
            df["Multiplier"] = 1.0
    except Exception:
        df["Multiplier"] = 1.0

    if "Secondary Position" in df.columns:
        df["Positions played"] = df["Position"].fillna("") + np.where(
            df["Secondary Position"].notna(), ", " + df["Secondary Position"].astype(str), ""
        )
    else:
        df["Positions played"] = df["Position"]

    df["Six-Group Position"] = df["Position"].apply(map_first_position_to_group)
        # ✅ Duplicate Centre Midfielders into Number 6 and Number 8
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

# ============================================================
# Position-specific ranking logic (with LFC variant + improved weighting logic)
# ============================================================
def compute_scores(df_all: pd.DataFrame, min_minutes: int = 600) -> pd.DataFrame:
    """
    Compute position-specific Z-scores and rankings using only relevant metrics per position.
    Matches the Radar page logic exactly.
    """
    pos_col = "Six-Group Position"
    if pos_col not in df_all.columns:
        df_all[pos_col] = np.nan

    df_all = df_all.copy()
    mins = pd.to_numeric(df_all.get("Minutes played", np.nan), errors="coerce").fillna(0)

    # --- Define position-specific metrics (copied from Radar page) ---
    position_metrics = {
        "Centre Back": [
            "NP Goals", "Passing%", "Pass OBV", "Pr. Long Balls", "UPr. Long Balls", "OBV", "Pr. Pass% Dif.",
            "PAdj Interceptions", "PAdj Tackles", "Dribbles Stopped%",
            "Defensive Actions", "Aggressive Actions", "Fouls", "Aerial Wins", "Aerial Win%",
        ],
        "Full Back": [
            "Passing%", "Pr. Pass% Dif.", "Successful Box Cross%", "Crossing%", "Deep Progressions",
            "Successful Dribbles", "Turnovers", "OBV", "Pass OBV",
            "Defensive Actions", "Aerial Win%", "PAdj Pressures",
            "PAdj Tack&Int", "Dribbles Stopped%", "Aggressive Actions", "Player Season Ball Recoveries 90"
        ],
        "Number 6": [
            "xGBuildup", "xG Assisted", "Passing%", "Deep Progressions", "Turnovers", "OBV", "Pass OBV", "Pr. Pass% Dif.",
            "PAdj Interceptions", "PAdj Tackles", "Dribbles Stopped%",
            "Aggressive Actions", "Aerial Win%", "Player Season Ball Recoveries 90", "Pressure Regains",
        ],
        "Number 8": [
            "xGBuildup", "xG Assisted", "Shots", "xG", "NP Goals",
            "Passing%", "Deep Progressions", "OP Passes Into Box", "Pass OBV", "OBV", "Deep Completions",
            "Pressure Regains", "PAdj Pressures", "Player Season Fhalf Ball Recoveries 90", "Aggressive Actions",
        ],
        "Winger": [
            "xG", "Shots", "xG/Shot", "Touches In Box", "OP xG Assisted", "NP Goals",
            "OP Passes Into Box", "Successful Box Cross%", "Passing%",
            "Successful Dribbles", "Turnovers", "OBV", "D&C OBV", "Fouls Won", "Deep Progressions",
            "Player Season Fhalf Pressures 90",
        ],
        "Striker": [
            "Aggressive Actions", "NP Goals", "xG", "Shots", "xG/Shot", "Goal Conversion%",
            "Touches In Box", "xG Assisted", "Fouls Won", "Deep Completions", "OP Key Passes",
            "Aerial Win%", "Aerial Wins", "Player Season Fhalf Pressures 90",
        ]
    }

    # --- Eligible baseline set (600+ minutes) ---
    eligible = df_all[mins >= min_minutes].copy()
    if eligible.empty:
        eligible = df_all.copy()

    # --- Initialize storage for Z-scores ---
    df_all["Avg Z Score"] = 0.0
    df_all["Weighted Z Score"] = 0.0
    df_all["LFC Weighted Z"] = 0.0
    df_all["Score (0–100)"] = 50.0
    df_all["LFC Score (0–100)"] = 50.0

    # --- Process each position separately ---
    for position, metrics in position_metrics.items():
        # Filter to players in this position
        pos_mask = df_all[pos_col] == position
        if not pos_mask.any():
            continue

        # Ensure all metrics exist
        for m in metrics:
            if m not in df_all.columns:
                df_all[m] = 0
            df_all[m] = pd.to_numeric(df_all[m], errors="coerce").fillna(0)

        # Get eligible players for this position (for baseline stats)
        eligible_pos = eligible[eligible[pos_col] == position]
        if eligible_pos.empty:
            eligible_pos = df_all[pos_mask].copy()

        # Compute mean/std from eligible players only
        baseline_stats = eligible_pos[metrics].agg(["mean", "std"]).fillna(0)
        baseline_stats.columns = baseline_stats.columns.map(lambda x: f"{x[0]}_{x[1]}")

        # Compute Z-scores for all players in this position
        raw_z = pd.DataFrame(index=df_all[pos_mask].index, columns=metrics, dtype=float)
        
        for m in metrics:
            mean_col = f"{m}_mean"
            std_col = f"{m}_std"
            
            if mean_col not in baseline_stats.index or std_col not in baseline_stats.index:
                raw_z[m] = 0
                continue
            
            mean_val = baseline_stats[mean_col]
            std_val = baseline_stats[std_col] if baseline_stats[std_col] != 0 else 1
            
            z = (df_all.loc[pos_mask, m] - mean_val) / std_val
            
            # Invert for "lower is better" metrics
            if m in LOWER_IS_BETTER:
                z *= -1
            
            raw_z[m] = z.fillna(0)

        # Average Z-score for this position
        avg_z = raw_z.mean(axis=1).fillna(0)
        df_all.loc[pos_mask, "Avg Z Score"] = avg_z

    # --- Apply league multipliers (same logic for all positions) ---
    mult = pd.to_numeric(df_all.get("Multiplier", 1.0), errors="coerce").fillna(1.0)
    avg_z = df_all["Avg Z Score"]

    df_all["Weighted Z Score"] = np.select(
        [avg_z > 0, avg_z < 0],
        [avg_z * mult, avg_z / mult],
        default=0.0
    )

    # --- LFC weighted variant (Scotland Premiership = 1.20) ---
    df_all["LFC Multiplier"] = mult.copy()
    df_all.loc[df_all["Competition_norm"] == "Scotland Premiership", "LFC Multiplier"] = 1.20
    lfc_mult = df_all["LFC Multiplier"]

    df_all["LFC Weighted Z"] = np.select(
        [avg_z > 0, avg_z < 0],
        [avg_z * lfc_mult, avg_z / lfc_mult],
        default=0.0
    )

    # --- Anchors per position (based on standard weighted scores) ---
    eligible = df_all[mins >= min_minutes].copy()
    if eligible.empty:
        eligible = df_all.copy()

    anchors = (
        eligible.groupby(pos_col, dropna=False)["Weighted Z Score"]
        .agg(_scale_min="min", _scale_max="max")
        .fillna(0)
    )

    if not anchors.empty:
        df_all = df_all.merge(anchors, left_on=pos_col, right_index=True, how="left", suffixes=('', '_anchor'))
    else:
        df_all["_scale_min"] = 0.0
        df_all["_scale_max"] = 1.0

    # --- Convert to 0–100 scale ---
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
        .fillna(50.0)
    )

    df_all.drop(columns=["_scale_min", "_scale_max"], inplace=True, errors="ignore")
    return df_all

# ============================================================
# Main UI
# ============================================================
try:
    df_all_raw = load_statsbomb(DATA_PATH)
    df_all_raw = add_age_column(df_all_raw)
    df_all = preprocess(df_all_raw)
    df_all = compute_scores(df_all, min_minutes=600)

    league_col = "Competition_norm"
    leagues = sorted(df_all[league_col].dropna().unique())
    selected_league = st.selectbox("Select League", leagues)

    clubs = sorted(df_all.loc[df_all[league_col] == selected_league, "Team"].dropna().unique())
    selected_club = st.selectbox("Select Club", clubs)

    df_team = df_all[(df_all[league_col] == selected_league) & (df_all["Team"] == selected_club)].copy()
    if df_team.empty:
        st.warning("No players found for this team.")
        st.stop()

    df_team["Rank in Team"] = df_team["Score (0–100)"].rank(ascending=False, method="min").astype(int)
    
    # ---------- Optional minutes filter (safe numeric version) ----------
    st.markdown("#### ⏱ Filter by Minutes Played (Display Only)")
    
    df_team["Minutes played"] = pd.to_numeric(df_team["Minutes played"], errors="coerce").fillna(0).astype(int)
    min_val = int(df_team["Minutes played"].min())
    max_val = int(df_team["Minutes played"].max())
    
    # Safe default clamp
    default_display_min = st.session_state.get("display_minutes_input", 600)
    if default_display_min > max_val:
        default_display_min = max_val
    
    # Use number input safely
    selected_min_display = st.number_input(
        "Show only players with at least this many minutes",
        min_value=min_val,
        max_value=max_val,
        value=default_display_min,
        step=50,
        key="display_minutes_input"
    )
    
    # Apply the filter
    df_team = df_team[df_team["Minutes played"] >= selected_min_display].copy()
    
    if df_team.empty:
        st.warning("No players available — try lowering your minimum minutes filter.")
        st.stop()

    # ---------- Team average scores ----------
    avg_score = df_team["Score (0–100)"].mean() if "Score (0–100)" in df_team.columns else np.nan
    avg_lfc = df_team["LFC Score (0–100)"].mean() if "LFC Score (0–100)" in df_team.columns else np.nan

    if not np.isnan(avg_score):
        if not np.isnan(avg_lfc):
            st.markdown(f"### {selected_club} ({selected_league}) — Average {avg_score:.1f} ({avg_lfc:.1f} LFC Score)")
        else:
            st.markdown(f"### {selected_club} ({selected_league}) — Average {avg_score:.1f}")
    else:
        st.markdown(f"### {selected_club} ({selected_league}) — No eligible players")

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
