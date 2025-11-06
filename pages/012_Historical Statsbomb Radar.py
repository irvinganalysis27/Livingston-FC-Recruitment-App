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

st.set_page_config(page_title="Historical Leagues - statsbombs radar", layout="centered")

# ---------- Auth ----------
if not check_password():
    st.stop()

# ---------- Branding ----------
show_branding()
st.title("Historical Leagues - statsbombs radar")

# ========== Helpers ==========
APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent

SIX_GROUPS = ["Full Back", "Centre Back", "Number 6", "Number 8", "Winger", "Striker"]

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

def map_first_position_to_group(primary_pos_cell) -> str:
    tok = _clean_pos_token(primary_pos_cell)
    return RAW_TO_SIX.get(tok, None)

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

def open_image(path: Path):
    try:
        return Image.open(path)
    except Exception:
        return None

# ========== Upload area ==========
st.markdown("### 📂 Upload a StatsBomb or Wyscout export file")
uploaded = st.file_uploader("Upload Excel or CSV file", type=["csv", "xlsx", "xls"])
if uploaded is None:
    st.info("Please upload your player data file to continue.")
    st.stop()

# Read file (no fallback)
try:
    if uploaded.name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

if df.empty:
    st.error("Uploaded file is empty.")
    st.stop()

# ========== Light preprocessing (names, positions, etc.) ==========
df.columns = (
    df.columns.astype(str)
      .str.strip()
      .str.replace(u"\xa0", " ", regex=False)
      .str.replace(r"\s+", " ", regex=True)
)

# Standard ID columns
rename_map = {}
if "Name" in df.columns:
    rename_map["Name"] = "Player"
if "Primary Position" in df.columns:
    rename_map["Primary Position"] = "Position"
if "Minutes" in df.columns:
    rename_map["Minutes"] = "Minutes played"
df.rename(columns=rename_map, inplace=True)

# Position mapping (no league logic)
df["Six-Group Position"] = df.get("Position", "").apply(map_first_position_to_group)

# Duplicate true central midfielders into 6 and 8
cm_mask = df["Six-Group Position"].eq("Centre Midfield")
if cm_mask.any():
    cm_as_6 = df.loc[cm_mask].copy()
    cm_as_6["Six-Group Position"] = "Number 6"
    cm_as_8 = df.loc[cm_mask].copy()
    cm_as_8["Six-Group Position"] = "Number 8"
    df = pd.concat([df, cm_as_6, cm_as_8], ignore_index=True)

# Age derivation if Birth Date exists
if "Birth Date" in df.columns and "Age" not in df.columns:
    today = datetime.today()
    dob = pd.to_datetime(df["Birth Date"], errors="coerce")
    df["Age"] = dob.apply(
        lambda d: today.year - d.year - ((today.month, today.day) < (d.month, d.day))
        if pd.notnull(d)
        else np.nan
    )

# Standard id columns
rename_map = {}
if "Name" in df.columns: rename_map["Name"] = "Player"
if "Primary Position" in df.columns: rename_map["Primary Position"] = "Position"
if "Minutes" in df.columns: rename_map["Minutes"] = "Minutes played"
df.rename(columns=rename_map, inplace=True)

# League column normalisation (no mapping needed)
if "Competition" in df.columns:
    df["Competition_norm"] = df["Competition"].astype(str).str.strip()
else:
    df["Competition_norm"] = np.nan

# Position mapping
df["Six-Group Position"] = df.get("Position", "").apply(map_first_position_to_group)

# Duplicate true central midfielders into 6 and 8
cm_mask = df["Six-Group Position"].eq("Centre Midfield")
if cm_mask.any():
    cm_as_6 = df.loc[cm_mask].copy(); cm_as_6["Six-Group Position"] = "Number 6"
    cm_as_8 = df.loc[cm_mask].copy(); cm_as_8["Six-Group Position"] = "Number 8"
    df = pd.concat([df, cm_as_6, cm_as_8], ignore_index=True)

# Age derivation if Birth Date exists
if "Birth Date" in df.columns and "Age" not in df.columns:
    today = datetime.today()
    dob = pd.to_datetime(df["Birth Date"], errors="coerce")
    df["Age"] = dob.apply(lambda d: today.year - d.year - ((today.month, today.day) < (d.month, d.day)) if pd.notnull(d) else np.nan)

# ========== Basic filters (same logic as Wyscout radar) ==========

# Minutes filter
minutes_col = "Minutes played"
if minutes_col not in df.columns:
    df[minutes_col] = np.nan
df["_minutes_numeric"] = pd.to_numeric(df[minutes_col], errors="coerce")

if "min_minutes" not in st.session_state:
    st.session_state.min_minutes = 600

# Minutes filter
minutes_col = "Minutes played"
if minutes_col not in df.columns:
    df[minutes_col] = np.nan
df["_minutes_numeric"] = pd.to_numeric(df[minutes_col], errors="coerce")
c1, c2 = st.columns(2)
with c1:
    if "min_minutes" not in st.session_state:
        st.session_state.min_minutes = 600
    st.session_state.min_minutes = st.number_input(
        "Minimum minutes to include", min_value=0, value=st.session_state.min_minutes, step=50, key="min_minutes_input"
    )
    df = df[df["_minutes_numeric"] >= st.session_state.min_minutes].copy()
    if df.empty:
        st.warning("No players meet the minutes threshold.")
        st.stop()

# Age filter
with c2:
    if "Age" in df.columns:
        df["_age_numeric"] = pd.to_numeric(df["Age"], errors="coerce")
        if df["_age_numeric"].notna().any():
            a_min, a_max = int(np.nanmin(df["_age_numeric"])), int(np.nanmax(df["_age_numeric"]))
            if "age_range" not in st.session_state:
                st.session_state.age_range = (a_min, a_max)
            st.session_state.age_range = st.slider("Age range to include", min_value=a_min, max_value=a_max, value=st.session_state.age_range, step=1, key="age_range_slider")
            lo, hi = st.session_state.age_range
            df = df[df["_age_numeric"].between(lo, hi)].copy()

st.caption(f"Players after filters: {len(df)}")

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

# Template chooser - snap to single group if only one selected
if "template_select" not in st.session_state:
    st.session_state.template_select = list(position_metrics.keys())[0]
if len(selected_groups) == 1 and selected_groups[0] in position_metrics:
    st.session_state.template_select = selected_groups[0]

template_names = list(position_metrics.keys())
if st.session_state.template_select not in template_names:
    st.session_state.template_select = template_names[0]
selected_position_template = st.selectbox("Radar Template", template_names, index=template_names.index(st.session_state.template_select), key="template_select", label_visibility="collapsed")

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
keep_cols = ["Player", "Team", "Team within selected timeframe", "Age", "Height", "Positions played", "Minutes played", "Six-Group Position"]
for c in keep_cols:
    if c not in df.columns: df[c] = np.nan

metrics_df = df[metrics].copy()
percentiles = (metrics_df.rank(pct=True) * 100).round(1)

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

# Chart
def plot_radial_bar_grouped(player_name, plot_df, metric_groups):
    import matplotlib.patches as mpatches
    from matplotlib import colormaps as mcm
    import matplotlib.colors as mcolors

    row_df = plot_df.loc[plot_df["Player"] == player_name]
    if row_df.empty:
        st.warning(f"No row for {player_name}")
        return
    row = row_df.iloc[0]

    # fixed order
    group_order = ["Possession", "Defensive", "Attacking", "Off The Ball", "Goalkeeping"]
    ordered_metrics = [m for g in group_order for m, gg in metric_groups.items() if gg == g]
    # take only those present
    valid_metrics = [m for m in ordered_metrics if m in plot_df.columns and f"{m} (percentile)" in plot_df.columns]
    if not valid_metrics:
        st.info("No valid metrics to plot.")
        return

    raw_vals = pd.to_numeric(row[valid_metrics], errors="coerce").fillna(0).to_numpy()
    pct_vals = pd.to_numeric(row[[f"{m} (percentile)" for m in valid_metrics]], errors="coerce").fillna(50).to_numpy()

    n = len(valid_metrics)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)

    fig, ax = plt.subplots(figsize=(10,10), subplot_kw=dict(polar=True))
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
    ax.set_ylim(0, 100); ax.set_yticklabels([]); ax.set_xticks([])
    ax.spines["polar"].set_visible(False)

    cmap = mcm.get_cmap("RdYlGn"); norm = mcolors.Normalize(vmin=0, vmax=100)
    bar_colors = [cmap(norm(v)) for v in pct_vals]

    ax.bar(angles, pct_vals, width=2*np.pi/n*0.85, color=bar_colors, edgecolor="black", linewidth=0.6, alpha=0.9)

    # raw values in middle ring
    for ang, raw in zip(angles, raw_vals):
        ax.text(ang, 50, f"{raw:.2f}", ha="center", va="center", fontsize=10, color="black", fontweight="bold")

    # labels
    for ang, m in zip(angles, valid_metrics):
        label = m.replace(" per 90","").replace(", %"," (%)")
        ax.text(ang, 108, label, ha="center", va="center", fontsize=10, color="black", fontweight="bold")

    # title
    role = row.get("Six-Group Position") or ""
    age = row.get("Age"); team = row.get("Team within selected timeframe") or row.get("Team") or ""
    mins = row.get("Minutes played")
    comp = row.get("Competition_norm") if "Competition_norm" in row.index else ""
    rank_v = int(row.get("Rank")) if pd.notnull(row.get("Rank")) else None
    z = float(row.get("Avg Z Score") or 0)

    top = " | ".join([x for x in [player_name, role, f"{int(age)} years old" if pd.notnull(age) else None] if x])
        parts = [
        team if team else "",
        comp if comp else "",
        f"{int(mins)} mins" if pd.notnull(mins) else "",
        f"Rank #{int(rank_v)}" if pd.notnull(rank_v) else "",
        f"Avg Z {z:.2f}" if pd.notnull(z) else "",
    ]
    parts = [str(p) for p in parts if p]  # ensure all strings, remove empties
    bottom = " | ".join(parts)

    ax.set_title(f"{top}\n{bottom}", color="black", size=22, pad=20, y=1.10)
    st.pyplot(fig, use_container_width=True)

# draw chart
plot_radial_bar_grouped(selected_player, plot_data, metric_groups)

# Ranking table (Avg Z only)
st.markdown("### Players Ranked by Z-Score")
cols_for_table = ["Player", "Positions played", "Age", "Team", "Team within selected timeframe",
                  "Minutes played", "Avg Z Score", "Rank"]
for c in cols_for_table:
    if c not in plot_data.columns: plot_data[c] = np.nan
table = (plot_data[cols_for_table]
         .sort_values(by="Avg Z Score", ascending=False)
         .reset_index(drop=True))
table[["Team","Team within selected timeframe"]] = table[["Team","Team within selected timeframe"]].fillna("N/A")
if "Age" in table.columns:
    table["Age"] = table["Age"].apply(lambda x: int(x) if pd.notnull(x) else x)
table.index = np.arange(1, len(table)+1); table.index.name = "Row"
st.dataframe(table, use_container_width=True)
