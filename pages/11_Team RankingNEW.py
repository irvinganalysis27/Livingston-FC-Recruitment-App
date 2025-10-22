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
# Page setup & access control
# ============================================================
st.set_page_config(page_title="Livingston FC Recruitment App — Team Rankings", layout="centered")

if not check_password():
    st.stop()

show_branding()
st.title("🏆 Team Player Rankings")

APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent
DATA_PATH = ROOT_DIR / "statsbomb_player_stats_clean.csv"
DB_PATH = APP_DIR / "favourites.db"

# ============================================================
# Favourites — local SQLite (simple + reliable)
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
    conn.execute(
        "INSERT OR REPLACE INTO favourites (player, team, league, position) VALUES (?,?,?,?)",
        (player, team, league, position)
    )
    conn.commit()
    conn.close()

def remove_favourite(player):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM favourites WHERE player=?", (player,))
    conn.commit()
    conn.close()

# ============================================================
# Robust data loading
# ============================================================
def _read_any(p: Path) -> pd.DataFrame:
    """Read CSV or Excel with safe fallbacks."""
    if p.suffix.lower() in {".xlsx", ".xls"}:
        try:
            import openpyxl  # excel reader
            return pd.read_excel(p, engine="openpyxl")
        except Exception:
            return pd.read_excel(p)  # fallback
    # CSV attempts (detect sep, encoding fallback)
    for kwargs in [
        dict(sep=None, engine="python"),
        dict(),
        dict(encoding="latin1"),
    ]:
        try:
            return pd.read_csv(p, **kwargs)
        except Exception:
            continue
    raise ValueError(f"Unsupported or unreadable file: {p.name}")

@st.cache_data(show_spinner=True)
def load_statsbomb(path: Path) -> pd.DataFrame:
    """Load single file or concatenate folder contents."""
    if path.is_file():
        df = _read_any(path)
    else:
        files = sorted(
            f for f in path.iterdir()
            if f.is_file() and f.suffix.lower() in {".csv", ".xlsx", ".xls", ""}
        )
        frames = []
        for f in files:
            try:
                frames.append(_read_any(f))
            except Exception:
                continue
        if not frames:
            raise FileNotFoundError(f"No readable player data files found under: {path}")
        df = pd.concat(frames, ignore_index=True, sort=False)

    # Clean column names early
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(u"\xa0", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )
    return df

def add_age_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add numeric Age from 'Birth Date' or 'birth_date' if present."""
    df = df.copy()
    today = datetime.today()
    birth_col = None
    for c in df.columns:
        if c.strip().lower() in {"birth date", "birth_date"}:
            birth_col = c
            break
    if birth_col:
        df["Age"] = pd.to_datetime(df[birth_col], errors="coerce").apply(
            lambda dob: today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            if pd.notna(dob) else np.nan
        )
    return df

# ============================================================
# Position mapping (to Six-Group) + basics
# ============================================================
RAW_TO_GROUP = {
    "LEFTBACK": "Full Back", "LEFTWINGBACK": "Full Back",
    "RIGHTBACK": "Full Back", "RIGHTWINGBACK": "Full Back",
    "CENTREBACK": "Centre Back", "LEFTCENTREBACK": "Centre Back", "RIGHTCENTREBACK": "Centre Back",
    "DEFENSIVEMIDFIELDER": "Number 6", "LEFTDEFENSIVEMIDFIELDER": "Number 6",
    "RIGHTDEFENSIVEMIDFIELDER": "Number 6", "CENTREDEFENSIVEMIDFIELDER": "Number 6",
    "CENTREMIDFIELDER": "Centre Midfield", "LEFTCENTREMIDFIELDER": "Centre Midfield", "RIGHTCENTREMIDFIELDER": "Centre Midfield",
    "CENTREATTACKINGMIDFIELDER": "Number 8", "RIGHTATTACKINGMIDFIELDER": "Number 8",
    "LEFTATTACKINGMIDFIELDER": "Number 8", "SECONDSTRIKER": "Number 8", "10": "Number 8",
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
# Preprocess (rename, competition_norm, multipliers, positions)
# ============================================================
@st.cache_data(show_spinner=True)
def preprocess(df: pd.DataFrame, root_dir: Path) -> pd.DataFrame:
    df = df.copy()

    # Rename to expected schema
    rename_map = {}
    if "Name" in df.columns: rename_map["Name"] = "Player"
    if "Primary Position" in df.columns: rename_map["Primary Position"] = "Position"
    if "Minutes" in df.columns: rename_map["Minutes"] = "Minutes played"
    df.rename(columns=rename_map, inplace=True)

    # Competition_norm
    if "Competition_norm" not in df.columns and "Competition" in df.columns:
        df["Competition_norm"] = df["Competition"].astype(str)
    elif "Competition_norm" not in df.columns:
        df["Competition_norm"] = np.nan

    # Multipliers (optional file: league_multipliers.xlsx)
    try:
        mult_df = pd.read_excel(root_dir / "league_multipliers.xlsx")
        # Handle flexible column naming
        cols = {c.lower().strip(): c for c in mult_df.columns}
        league_col = cols.get("league", None) or cols.get("competition", None)
        mult_col = cols.get("multiplier", None)
        if league_col and mult_col:
            mult_df = mult_df[[league_col, mult_col]].copy()
            mult_df.columns = ["League", "Multiplier"]
            df = df.merge(mult_df, left_on="Competition_norm", right_on="League", how="left")
            df["Multiplier"] = pd.to_numeric(df["Multiplier"], errors="coerce").fillna(1.0)
            df.drop(columns=["League"], inplace=True, errors="ignore")
        else:
            df["Multiplier"] = 1.0
    except Exception:
        df["Multiplier"] = 1.0

    # Positions played
    if "Secondary Position" in df.columns:
        df["Positions played"] = df["Position"].fillna("").astype(str) + np.where(
            df["Secondary Position"].notna() & (df["Secondary Position"].astype(str) != ""),
            ", " + df["Secondary Position"].astype(str),
            ""
        )
    else:
        df["Positions played"] = df.get("Position", np.nan)

    # Six-Group Position mapping (with Centre Midfield duplication to 6 & 8 once per player-team)
    df["Six-Group Position"] = df.get("Position", np.nan).apply(map_first_position_to_group)

    if "Six-Group Position" in df.columns:
        cm_mask = df["Six-Group Position"].eq("Centre Midfield")
        if cm_mask.any():
            cm_rows = (
                df.loc[cm_mask, ["Player", "Team", "Six-Group Position"]]
                .drop_duplicates(subset=["Player", "Team"])
            )
            if not cm_rows.empty:
                cm_as_6 = df.loc[
                    df["Player"].isin(cm_rows["Player"]) &
                    df["Team"].isin(cm_rows["Team"]) &
                    cm_mask
                ].copy()
                cm_as_8 = cm_as_6.copy()
                cm_as_6["Six-Group Position"] = "Number 6"
                cm_as_8["Six-Group Position"] = "Number 8"

                already_6_8 = df[
                    (df["Six-Group Position"].isin(["Number 6", "Number 8"])) &
                    (df["Player"].isin(cm_rows["Player"]))
                ]
                new_rows = pd.concat([cm_as_6, cm_as_8], ignore_index=True)
                new_rows = new_rows[
                    ~new_rows.set_index(["Player", "Team", "Six-Group Position"]).index.isin(
                        already_6_8.set_index(["Player", "Team", "Six-Group Position"]).index
                    )
                ]
                df = pd.concat([df, new_rows], ignore_index=True)

    # Ensure Minutes played exists + numeric
    if "Minutes played" not in df.columns:
        df["Minutes played"] = np.nan
    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")

    return df

# ============================================================
# Scoring logic (Z-scores, Weighted, LFC variant, 0–100 scaling)
# ============================================================
@st.cache_data(show_spinner=True)
def compute_scores(df_all_in: pd.DataFrame, min_minutes: int = 600) -> pd.DataFrame:
    df_all = df_all_in.copy()
    pos_col = "Six-Group Position"
    if pos_col not in df_all.columns:
        df_all[pos_col] = np.nan

    mins = pd.to_numeric(df_all.get("Minutes played", np.nan), errors="coerce").fillna(0)

    # Eligible baseline set
    eligible = df_all[mins >= min_minutes].copy()
    if eligible.empty:
        eligible = df_all.copy()

    # Numeric columns & baseline stats per position (mean/std)
    num_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
    # don't include columns not true performance metrics
    ignore_cols = {"Age", "Height", "Minutes played"}
    metric_cols = [c for c in num_cols if c not in ignore_cols]

    baseline_stats = eligible.groupby(pos_col)[metric_cols].agg(["mean", "std"]).fillna(0)
    baseline_stats.columns = baseline_stats.columns.map("_".join)

    # Raw Z for each metric per position (invert lower-is-better)
    raw_z = pd.DataFrame(index=df_all.index, columns=metric_cols, dtype=float)
    for m in metric_cols:
        mean_col = f"{m}_mean"
        std_col = f"{m}_std"
        if mean_col not in baseline_stats.columns or std_col not in baseline_stats.columns:
            raw_z[m] = 0.0
            continue
        mean_vals = df_all[pos_col].map(baseline_stats[mean_col])
        std_vals = df_all[pos_col].map(baseline_stats[std_col].replace(0, 1))
        z = (pd.to_numeric(df_all[m], errors="coerce").fillna(0) - mean_vals) / std_vals
        if m in LOWER_IS_BETTER:
            z *= -1
        raw_z[m] = z.fillna(0)

    # Average Z and League-weighted logic
    df_all["Avg Z Score"] = raw_z.mean(axis=1).fillna(0)
    mult = pd.to_numeric(df_all.get("Multiplier", 1.0), errors="coerce").fillna(1.0)
    avg_z = df_all["Avg Z Score"]

    df_all["Weighted Z Score"] = np.select(
        [avg_z > 0, avg_z < 0],
        [avg_z * mult, avg_z / mult],
        default=0.0
    )

    # LFC variant: Scotland Premiership boost to 1.20
    lfc_mult = mult.copy()
    df_all["LFC Multiplier"] = lfc_mult
    df_all.loc[df_all["Competition_norm"] == "Scotland Premiership", "LFC Multiplier"] = 1.20
    lfc_mult = pd.to_numeric(df_all["LFC Multiplier"], errors="coerce").fillna(1.0)

    df_all["LFC Weighted Z"] = np.select(
        [avg_z > 0, avg_z < 0],
        [avg_z * lfc_mult, avg_z / lfc_mult],
        default=0.0
    )

    # Anchors per position for scaling 0–100 (using standard Weighted Z)
    eligible2 = df_all[mins >= min_minutes].copy()
    if eligible2.empty:
        eligible2 = df_all.copy()

    anchors = (
        eligible2.groupby(pos_col, dropna=False)["Weighted Z Score"]
        .agg(_scale_min="min", _scale_max="max")
        .fillna(0)
    )
    if not anchors.empty:
        df_all = df_all.merge(anchors, left_on=pos_col, right_index=True, how="left")
    else:
        df_all["_scale_min"] = 0.0
        df_all["_scale_max"] = 1.0

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

    df_all.drop(columns=["_scale_min", "_scale_max"], inplace=True, errors="ignore")
    return df_all

# ============================================================
# UI — League → Club → Table + Favourites
# ============================================================
try:
    df_all_raw = load_statsbomb(DATA_PATH)
    df_all_raw = add_age_column(df_all_raw)
    df_all = preprocess(df_all_raw, ROOT_DIR)
    df_all = compute_scores(df_all, min_minutes=600)

    league_col = "Competition_norm"
    if league_col not in df_all.columns:
        st.error("❌ Expected a league/competition column after preprocessing.")
        st.stop()

    # League & Club selectors
    leagues = sorted(x for x in df_all[league_col].dropna().unique() if str(x).strip())
    if not leagues:
        st.warning("No leagues found in the dataset.")
        st.stop()
    selected_league = st.selectbox("Select League", leagues, index=0)

    clubs = sorted(
        x for x in df_all.loc[df_all[league_col] == selected_league, "Team"].dropna().unique()
        if str(x).strip()
    )
    if not clubs:
        st.warning("No clubs found in this league.")
        st.stop()
    selected_club = st.selectbox("Select Club", clubs, index=0)

    # Team slice
    df_team = df_all[(df_all[league_col] == selected_league) & (df_all["Team"] == selected_club)].copy()
    if df_team.empty:
        st.warning("No players found for this team.")
        st.stop()

    # Display-only minutes filter (does not affect baseline computation)
    df_team["Minutes played"] = pd.to_numeric(df_team["Minutes played"], errors="coerce").fillna(0).astype(int)
    st.markdown("#### ⏱ Filter by Minutes Played (Display Only)")
    min_val = int(df_team["Minutes played"].min())
    max_val = int(df_team["Minutes played"].max())
    default_display_min = st.session_state.get("display_minutes_input", min(600, max_val))
    default_display_min = int(np.clip(default_display_min, min_val, max_val))
    selected_min_display = st.number_input(
        "Show only players with at least this many minutes",
        min_value=min_val,
        max_value=max_val,
        value=default_display_min,
        step=50,
        key="display_minutes_input"
    )
    df_team = df_team[df_team["Minutes played"] >= selected_min_display].copy()
    if df_team.empty:
        st.warning("No players available — try lowering your minimum minutes filter.")
        st.stop()

    # Rank within team by standard score
    if "Score (0–100)" in df_team.columns:
        df_team["Rank in Team"] = df_team["Score (0–100)"].rank(ascending=False, method="min").astype(int)
    else:
        df_team["Rank in Team"] = np.nan

    # Team averages (for context)
    avg_score = df_team["Score (0–100)"].mean() if "Score (0–100)" in df_team.columns else np.nan
    avg_lfc = df_team["LFC Score (0–100)"].mean() if "LFC Score (0–100)" in df_team.columns else np.nan
    if pd.notna(avg_score) and pd.notna(avg_lfc):
        st.markdown(f"### {selected_club} ({selected_league}) — Average {avg_score:.1f} ({avg_lfc:.1f} LFC Score)")
    elif pd.notna(avg_score):
        st.markdown(f"### {selected_club} ({selected_league}) — Average {avg_score:.1f}")
    else:
        st.markdown(f"### {selected_club} ({selected_league})")

    # Build display table
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

    # Nice numeric formatting
    z_ranking["Age"] = pd.to_numeric(z_ranking["Age"], errors="coerce").round(0)
    z_ranking["Minutes played"] = pd.to_numeric(z_ranking["Minutes played"], errors="coerce").fillna(0).astype(int)
    z_ranking["League Weight"] = pd.to_numeric(z_ranking["League Weight"], errors="coerce").fillna(1.0).round(3)

    # Favourites integration
    favs_in_db = {row[0] for row in get_favourites()}
    z_ranking["⭐ Favourite"] = z_ranking["Player"].isin(favs_in_db)

    # Sort by primary score then by team rank
    if "Score (0–100)" in z_ranking.columns:
        z_ranking.sort_values(["Score (0–100)", "Z Weighted"], ascending=[False, False], inplace=True)

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
        use_container_width=True,
    )

    # Apply favourites changes
    for _, r in edited.iterrows():
        p = r.get("Player")
        if not p:
            continue
        is_star = bool(r.get("⭐ Favourite", False))
        if is_star and p not in favs_in_db:
            add_favourite(p, r.get("Team"), r.get("League"), r.get("Positions played"))
        elif (not is_star) and p in favs_in_db:
            remove_favourite(p)

    # Optional export
    st.download_button(
        "Download table as CSV",
        data=edited.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_club.replace(' ','_')}_rankings.csv",
        mime="text/csv",
    )

except Exception as e:
    st.error(f"❌ Could not load data: {e}")
