import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import re

import os
print("Current working directory:", os.getcwd())
print("Files in current folder:", os.listdir(APP_DIR))

# ========= PAGE CONFIG =========
st.set_page_config(page_title="SkillCorner Physical Radar", layout="centered")
st.title("📈 Physical Radar (SkillCorner)")

# ========= WHERE IS YOUR CSV? =========
# Point this to your SkillCorner export. Example: data/SkillCorner-2025-10-18.csv
APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "SkillCorner-2025-10-18.csv"

# ========= METRICS (fixed set for this page) =========
RADAR_METRICS = [
    "Distance P90",
    "M/min P90",
    "Running Distance P90",
    "HSR Distance P90",
    "Sprint Count P90",
    "High Acceleration Count P90",
    "High Deceleration Count P90",
    "PSV-99",
    "HI Distance P90",
]

# Grouping for colour/legend on the radar
METRIC_GROUPS = {
    "Distance P90": "Work Rate",
    "M/min P90": "Work Rate",

    "Running Distance P90": "Running Load",
    "HSR Distance P90": "Running Load",

    "Sprint Count P90": "Explosiveness",
    "High Acceleration Count P90": "Explosiveness",
    "High Deceleration Count P90": "Explosiveness",

    "PSV-99": "Top Speed",

    "HI Distance P90": "Intensity",
}

GROUP_COLOURS = {
    "Work Rate": "seagreen",
    "Running Load": "royalblue",
    "Explosiveness": "crimson",
    "Top Speed": "darkorange",
    "Intensity": "purple",
}

# ========= UTILS =========
def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(u"\xa0", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )
    return df

def _detect_and_read_csv(path: Path) -> pd.DataFrame:
    """
    Read a messy CSV/TSV robustly (auto-detect delimiter, skip bad lines).
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Try common strategies
    trials = [
        dict(sep=None, engine="python", on_bad_lines="skip"),  # auto-detect
        dict(sep=",", engine="python", on_bad_lines="skip"),
        dict(sep="\t", engine="python", on_bad_lines="skip"),
    ]
    for kwargs in trials:
        try:
            df = pd.read_csv(path, **kwargs)
            if df.shape[1] == 1:
                # Sometimes quoted CSV still collapses; try again with stricter parsing
                continue
            return df
        except Exception:
            continue
    # Last resort: read whole file then let pandas guess again
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    from io import StringIO
    return pd.read_csv(StringIO(text), sep=None, engine="python", on_bad_lines="skip")

def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def pct_rank(series: pd.Series, lower_is_better: bool = False) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    r = s.rank(pct=True, ascending=True)
    if lower_is_better:
        r = 1.0 - r
    return (r * 100.0).round(1)

# ========= CACHE LOAD =========
@st.cache_data(ttl=86400, show_spinner=True)  # 24h cache
def load_skillcorner_csv(path_str: str):
    path = Path(path_str)
    df = _detect_and_read_csv(path)
    df = _clean_columns(df)

    # Expect these columns (as per your sample)
    expected_cols = [
        "Player", "Short Name", "Player ID", "Birthdate",
        "Team", "Team ID",
        "Match", "Match ID", "Date",
        "Competition", "Competition ID",
        "Season", "Season ID", "Competition Edition ID",
        "Position", "Position Group",
        "Minutes", "Physical Check Passed",
        "Distance", "M/min", "Running Distance", "HSR Distance", "HSR Count",
        "Sprint Distance", "Sprint Count",
        "HI Distance", "HI Count",
        "Medium Acceleration Count", "High Acceleration Count",
        "Medium Deceleration Count", "High Deceleration Count",
        "Explosive Acceleration to HSR Count", "Time to HSR",
        "Explosive Acceleration to Sprint Count", "Time to Sprint",
        "PSV-99",
        # Per-90 block you showed:
        "Distance P90", "M/min P90", "Running Distance P90", "HSR Distance P90",
        "HSR Count P90", "Sprint Distance P90", "Sprint Count P90",
        "HI Distance P90", "HI Count P90",
        "Medium Acceleration Count P90", "High Acceleration Count P90",
        "Medium Deceleration Count P90", "High Deceleration Count P90",
        "Explosive Acceleration to HSR Count P90", "Explosive Acceleration to Sprint Count P90",
    ]

    # Create any missing expected columns so downstream code doesn't explode
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Basic type cleaning
    numeric_cols = list({
        "Minutes",
        "Distance", "M/min", "Running Distance", "HSR Distance", "HSR Count",
        "Sprint Distance", "Sprint Count",
        "HI Distance", "HI Count",
        "Medium Acceleration Count", "High Acceleration Count",
        "Medium Deceleration Count", "High Deceleration Count",
        "Explosive Acceleration to HSR Count", "Time to HSR",
        "Explosive Acceleration to Sprint Count", "Time to Sprint",
        "PSV-99",
        "Distance P90", "M/min P90", "Running Distance P90", "HSR Distance P90",
        "HSR Count P90", "Sprint Distance P90", "Sprint Count P90",
        "HI Distance P90", "HI Count P90",
        "Medium Acceleration Count P90", "High Acceleration Count P90",
        "Medium Deceleration Count P90", "High Deceleration Count P90",
        "Explosive Acceleration to HSR Count P90", "Explosive Acceleration to Sprint Count P90",
    })
    df = _safe_numeric(df, numeric_cols)

    # Parse dates if present
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Backup (optional)
    backup_dir = Path("data")
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / "skillcorner_physical_backup.csv"
    try:
        df.to_csv(backup_path, index=False)
    except Exception as e:
        st.info(f"(Optional) Could not write backup CSV: {e}")

    return df

# ========= LOAD DATA =========
st.caption(f"Source file: `{DATA_PATH}`")
try:
    df_raw = load_skillcorner_csv(str(DATA_PATH))
except Exception as e:
    st.error(f"❌ Failed to load CSV: {e}")
    st.stop()

if df_raw.empty:
    st.error("❌ No rows in the CSV.")
    st.stop()

with st.expander("🔎 Raw columns returned (first 10 rows)", expanded=False):
    st.write(df_raw.columns.tolist())
    st.dataframe(df_raw.head(10), use_container_width=True)

# ========= AGGREGATE TO PLAYER-LEVEL =========
# We’ll aggregate across matches: mean of P90 metrics; PSV-99 as max; Minutes as sum
agg_spec = {m: "mean" for m in RADAR_METRICS}
agg_spec["PSV-99"] = "max"  # sensible: peak speed as max observed
agg_spec["Minutes"] = "sum"
# Keep some identity fields by taking the most recent non-null (via last valid match after sort)
id_cols = ["Player", "Short Name", "Player ID", "Team", "Competition", "Season", "Position Group"]

df_sorted = df_raw.sort_values(["Player", "Date"], ascending=[True, True]).copy()

# Select only columns we need to avoid surprises
use_cols = list(set(id_cols + ["Date", "Minutes"] + RADAR_METRICS + ["PSV-99"]))
use_cols = [c for c in use_cols if c in df_sorted.columns]
df_use = df_sorted[use_cols].copy()

def _last_non_null(s: pd.Series):
    v = s.dropna()
    return v.iloc[-1] if len(v) else np.nan

groupers = ["Player", "Position Group"]
df_player = (
    df_use.groupby(groupers, dropna=False)
          .agg({**agg_spec,
                "Team": _last_non_null,
                "Competition": _last_non_null,
                "Season": _last_non_null})
          .reset_index()
)

# ========= FILTERS =========
st.markdown("#### Filters")
# League
leagues = (
    pd.Series(df_player["Competition"], dtype="string")
      .dropna().str.strip()
      .replace({"nan": pd.NA})
      .dropna().unique().tolist()
)
leagues = sorted(leagues)
default_leagues = leagues  # start with all selected
selected_leagues = st.multiselect("Leagues", options=leagues, default=default_leagues, key="sc_league_sel")

# Position Group (use what SkillCorner gives you)
pos_groups = (
    pd.Series(df_player["Position Group"], dtype="string")
      .dropna().str.strip()
      .replace({"nan": pd.NA})
      .dropna().unique().tolist()
)
pos_groups = sorted(pos_groups)
selected_pos_groups = st.multiselect("Position Groups", options=pos_groups, default=pos_groups, key="sc_pos_sel")

# Minutes threshold
min_minutes = st.number_input("Minimum total minutes", min_value=0, value=600, step=60)

# Apply filters
df = df_player.copy()
if selected_leagues:
    df = df[df["Competition"].isin(selected_leagues)]
if selected_pos_groups:
    df = df[df["Position Group"].isin(selected_pos_groups)]
df = df[df["Minutes"] >= min_minutes]

st.caption(f"Players after filters: **{len(df)}**")

if df.empty:
    st.warning("No players match the current filters. Adjust and try again.")
    st.stop()

# ========= PERCENTILES (within current selection by league OR pooled) =========
compute_within_league = st.checkbox("Percentiles within each league (Competition)", value=True)

percentile_df = pd.DataFrame(index=df.index, columns=RADAR_METRICS, dtype=float)
if compute_within_league and "Competition" in df.columns:
    for m in RADAR_METRICS:
        try:
            percentile_df[m] = (
                df.groupby("Competition", group_keys=False)[m]
                  .apply(lambda s: pct_rank(s, lower_is_better=False))
            )
        except Exception as e:
            print(f"[DEBUG] Percentile failed for {m}: {e}")
            percentile_df[m] = 50.0
else:
    for m in RADAR_METRICS:
        try:
            percentile_df[m] = pct_rank(df[m], lower_is_better=False)
        except Exception as e:
            print(f"[DEBUG] Percentile failed for {m}: {e}")
            percentile_df[m] = 50.0

percentile_df = percentile_df.fillna(50.0).round(1)

# Composite score for ranking (simple mean of percentiles)
df["_score_0_100"] = percentile_df.mean(axis=1).round(1)

# ========= PLAYER SELECT =========
players = df["Player"].dropna().unique().tolist()
players = sorted(players)

# Keep last selection stable
if "sc_selected_player" not in st.session_state or st.session_state.sc_selected_player not in players:
    st.session_state.sc_selected_player = players[0]

selected_player = st.selectbox("Choose a player", players, index=players.index(st.session_state.sc_selected_player))
st.session_state.sc_selected_player = selected_player

# ========= RADAR PLOT =========
def plot_radial_bar_grouped(player_name: str):
    import matplotlib.colors as mcolors
    from matplotlib import colormaps as mcm

    row_df = df.loc[df["Player"] == player_name]
    if row_df.empty:
        st.error(f"No player named '{player_name}' in current selection.")
        return
    row = row_df.iloc[0]
    pcts = percentile_df.loc[row_df.index[0], RADAR_METRICS].to_numpy(dtype=float)
    raws = df.loc[row_df.index[0], RADAR_METRICS].to_numpy(dtype=float)

    n = len(RADAR_METRICS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    cmap = mcm.get_cmap("RdYlGn")
    norm = mcolors.Normalize(vmin=0, vmax=100)
    bar_colors = [cmap(norm(v)) for v in pcts]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)

    ax.bar(angles, pcts, width=2 * np.pi / n * 0.85, color=bar_colors, edgecolor="black", linewidth=0.6, alpha=0.9)

    # Raw values in the middle ring
    for ang, raw_val in zip(angles, raws):
        txt = f"{raw_val:.1f}" if np.isfinite(raw_val) else "-"
        ax.text(ang, 50, txt, ha="center", va="center", color="black", fontsize=10, fontweight="bold")

    # Labels, coloured by group
    for ang, m in zip(angles, RADAR_METRICS):
        label = m
        group = METRIC_GROUPS.get(m, "Other")
        color = GROUP_COLOURS.get(group, "black")
        ax.text(ang, 108, label, ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    # Legend
    present_groups = list(dict.fromkeys([METRIC_GROUPS.get(m, "Other") for m in RADAR_METRICS]))
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=GROUP_COLOURS.get(g, "grey"), label=g) for g in present_groups]
    if patches:
        fig.subplots_adjust(top=0.86, bottom=0.08)
        ax.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.06),
                  ncol=min(len(patches), 4), frameon=False)

    # Title lines
    team = str(row.get("Team") or "")
    comp = str(row.get("Competition") or "")
    mins = row.get("Minutes")
    posg = str(row.get("Position Group") or "")

    line1 = " | ".join([x for x in [player_name, posg] if x])
    bottom_parts = []
    if team: bottom_parts.append(team)
    if comp: bottom_parts.append(comp)
    if pd.notnull(mins): bottom_parts.append(f"{int(mins)} mins")
    bottom_parts.append(f"{float(row.get('_score_0_100', np.nan)):.0f}/100")
    line2 = " | ".join(bottom_parts)

    ax.set_title(f"{line1}\n{line2}", color="black", size=22, pad=20, y=1.10)
    st.pyplot(fig, use_container_width=True)

# Plot
plot_radial_bar_grouped(selected_player)

# ========= RANKING TABLE =========
st.markdown("### Players Ranked by Physical Composite (0–100)")
rank_cols = ["Player", "Team", "Competition", "Position Group", "Minutes", "_score_0_100"] + RADAR_METRICS
table = df[rank_cols].copy()
table = table.sort_values("_score_0_100", ascending=False).reset_index(drop=True)
table.index = np.arange(1, len(table) + 1)
table.index.name = "Rank"
st.dataframe(table, use_container_width=True)
