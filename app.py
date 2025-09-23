import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

APP_DIR = Path(__file__).parent
ASSETS_DIR = APP_DIR / "assets"

def open_image(path: Path):
    try:
        return Image.open(path)
    except Exception:
        return None

# --- Basic password protection ---
PASSWORD = "Livi2025"

st.set_page_config(page_title="Livingston FC Recruitment App", layout="centered")

# ---------- Club Branding Row ----------
left, mid, right = st.columns([1, 6, 1])

logo_path = ASSETS_DIR / "Livingston_FC_club_badge_new.png"   # case-sensitive
logo = open_image(logo_path)

with left:
    if logo:
        st.image(logo, use_container_width=True)

with mid:
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1>Livingston FC Recruitment<br>App</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

with right:
    if logo:
        st.image(logo, use_container_width=True)

# Ask for password
pwd = st.text_input("Enter password:", type="password")
if pwd != PASSWORD:
    st.warning("Please enter the correct password to access the app.")
    st.stop()

# ========== Role groups shown in filters ==========
SIX_GROUPS = [
    "Goalkeeper",
    "Full Back",
    "Centre Back",
    "Number 6",
    "Number 8",
    "Winger",
    "Striker"
]

# ========== Position → group mapping for NEW PROVIDER labels ==========
def _clean_pos_token(tok: str) -> str:
    if pd.isna(tok):
        return ""
    t = str(tok).upper().strip()
    t = re.sub(r"[.\-_/]", " ", t)
    t = re.sub(r"\s+", "", t)
    return t

RAW_TO_SIX = {
    # GK
    "GOALKEEPER": "Goalkeeper",

    # Full backs & wing backs
    "RIGHTBACK": "Full Back",
    "LEFTBACK": "Full Back",
    "RIGHTWINGBACK": "Full Back",
    "LEFTWINGBACK": "Full Back",

    # Centre backs
    "RIGHTCENTREBACK": "Centre Back",
    "LEFTCENTREBACK": "Centre Back",
    "CENTREBACK": "Centre Back",

    # Centre mid (generic) → we’ll duplicate into 6 & 8 later
    "CENTREMIDFIELDER": "Centre Midfield",
    "RIGHTCENTREMIDFIELDER": "Centre Midfield",
    "LEFTCENTREMIDFIELDER": "Centre Midfield",

    # Defensive mids → 6
    "DEFENSIVEMIDFIELDER": "Number 6",
    "RIGHTDEFENSIVEMIDFIELDER": "Number 6",
    "LEFTDEFENSIVEMIDFIELDER": "Number 6",

    # Attacking mids / 10 → 8
    "CENTREATTACKINGMIDFIELDER": "Number 8",
    "ATTACKINGMIDFIELDER": "Number 8",
    "SECONDSTRIKER": "Number 8",
    "10": "Number 8",

    # Wingers / wide mids
    "RIGHTWING": "Winger",
    "LEFTWING": "Winger",
    "RIGHTMIDFIELDER": "Winger",
    "LEFTMIDFIELDER": "Winger",

    # Strikers
    "CENTREFORWARD": "Striker",
    "RIGHTCENTREFORWARD": "Striker",
    "LEFTCENTREFORWARD": "Striker",
}

def parse_first_position(cell) -> str:
    # We’ll use PRIMARY position only for mapping; UI will show both
    if pd.isna(cell):
        return ""
    return _clean_pos_token(str(cell))

def map_first_position_to_group(primary_pos_cell) -> str:
    tok = parse_first_position(primary_pos_cell)
    return RAW_TO_SIX.get(tok, "Winger")  # safe default

# ========== Default template mapping ==========
DEFAULT_TEMPLATE = {
    "Goalkeeper": "Goalkeeper",
    "Full Back": "Full Back",
    "Centre Back": "Centre Back",
    "Number 6": "Number 6",
    "Number 8": "Number 8",
    "Winger": "Winger",
    "Striker": "Striker"
}

# ========== Radar metric sets (NEW PROVIDER NAMES ONLY) ==========
position_metrics = {
    "Goalkeeper": {
        "metrics": [
            # Keepers (retain only if your new file has them; otherwise they'll show as 0s)
            "Goals Conceded", "PSxG Faced", "GSAA", "Save%", "xSv%", "Shot Stopping%",
            "Shots Faced", "Shots Faced OT%",
            "Pass into Danger%", "Pass into Pressure%",
            "Positive Outcome%", "Goalkeeper OBV"
        ],
        "groups": {
            "Goals Conceded": "Goalkeeping",
            "PSxG Faced": "Goalkeeping",
            "GSAA": "Goalkeeping",
            "Save%": "Goalkeeping",
            "xSv%": "Goalkeeping",
            "Shot Stopping%": "Goalkeeping",
            "Shots Faced": "Goalkeeping",
            "Shots Faced OT%": "Goalkeeping",
            "Pass into Danger%": "Possession",
            "Pass into Pressure%": "Possession",
            "Positive Outcome%": "Goalkeeping",
            "Goalkeeper OBV": "Goalkeeping",
        }
    },

    "Centre Back": {
        "metrics": [
            "PAdj Interceptions", "PAdj Tackles", "Tack/DP%",
            "Defensive Actions", "Ball Recoveries",
            "Aerial Win%", "Aerial Wins",
            "Passing%", "Pass OBV",
            "UPr. Long Balls",
            "Pressures", "PAdj Pressures"
        ],
        "groups": {
            "PAdj Interceptions": "Defensive",
            "PAdj Tackles": "Defensive",
            "Tack/DP%": "Defensive",
            "Defensive Actions": "Defensive",
            "Ball Recoveries": "Defensive",
            "Aerial Win%": "Defensive",
            "Aerial Wins": "Defensive",
            "Passing%": "Possession",
            "Pass OBV": "Possession",
            "UPr. Long Balls": "Possession",
            "Pressures": "Off The Ball",
            "PAdj Pressures": "Off The Ball",
        }
    },

    "Full Back": {
        "metrics": [
            "Pass OBV", "Passing%", "OP Passes Into Box",
            "Deep Progressions", "Deep Completions",
            "Successful Dribbles", "Dribbles",
            "Turnovers",
            "Defensive Actions",
            "Aerial Win%", "Aerial Wins",
            "Pressures", "PAdj Pressures",
            "Tack/DP%"
        ],
        "groups": {
            "Pass OBV": "Possession",
            "Passing%": "Possession",
            "OP Passes Into Box": "Possession",
            "Deep Progressions": "Possession",
            "Deep Completions": "Possession",
            "Successful Dribbles": "Possession",
            "Dribbles": "Possession",
            "Turnovers": "Possession",
            "Defensive Actions": "Defensive",
            "Aerial Win%": "Defensive",
            "Aerial Wins": "Defensive",
            "Pressures": "Off The Ball",
            "PAdj Pressures": "Off The Ball",
            "Tack/DP%": "Defensive",
        }
    },

    "Number 6": {
        "metrics": [
            "Passing%", "Deep Progressions", "Pass OBV",
            "PAdj Interceptions", "PAdj Tack&Int", "Ball Recoveries",
            "Tack/DP%", "Aggressive Actions",
            "Aerial Win%", "Aerial Wins",
            "Turnovers",
            "Pressures", "PAdj Pressures"
        ],
        "groups": {
            "Passing%": "Possession",
            "Deep Progressions": "Possession",
            "Pass OBV": "Possession",
            "PAdj Interceptions": "Defensive",
            "PAdj Tack&Int": "Defensive",
            "Ball Recoveries": "Defensive",
            "Tack/DP%": "Defensive",
            "Aggressive Actions": "Off The Ball",
            "Aerial Win%": "Defensive",
            "Aerial Wins": "Defensive",
            "Turnovers": "Possession",
            "Pressures": "Off The Ball",
            "PAdj Pressures": "Off The Ball",
        }
    },

    "Number 8": {
        "metrics": [
            "Passing%", "Deep Progressions",
            "xGChain", "xGBuildup",
            "xG Assisted", "OP Passes Into Box", "Pass OBV",
            "xG", "Shots", "xG/Shot",
            "Pressure Regains", "Aggressive Actions"
        ],
        "groups": {
            "Passing%": "Possession",
            "Deep Progressions": "Possession",
            "xGChain": "Attacking",
            "xGBuildup": "Attacking",
            "xG Assisted": "Attacking",
            "OP Passes Into Box": "Possession",
            "Pass OBV": "Possession",
            "xG": "Attacking",
            "Shots": "Attacking",
            "xG/Shot": "Attacking",
            "Pressure Regains": "Off The Ball",
            "Aggressive Actions": "Off The Ball",
        }
    },

    "Winger": {
        "metrics": [
            "NP Goals", "xG", "Shots", "xG/Shot",
            "xG Assisted", "Key Passes",
            "Successful Crosses", "Successful Box Cross%", "Crossing%",
            "Dribbles", "Successful Dribbles",
            "Fouls Won",
            "OP Passes Into Box", "Deep Completions"
        ],
        "groups": {
            "NP Goals": "Attacking",
            "xG": "Attacking",
            "Shots": "Attacking",
            "xG/Shot": "Attacking",
            "xG Assisted": "Attacking",
            "Key Passes": "Attacking",
            "Successful Crosses": "Possession",
            "Successful Box Cross%": "Possession",
            "Crossing%": "Possession",
            "Dribbles": "Possession",
            "Successful Dribbles": "Possession",
            "Fouls Won": "Possession",
            "OP Passes Into Box": "Possession",
            "Deep Completions": "Possession",
        }
    },

    "Striker": {
        "metrics": [
            "Pressure Regains",
            "All Goals", "Penalty Goals", "xG", "Shots", "xG/Shot",
            "Shot Touch%", "Touches In Box",
            "xG Assisted",
            "Aerial Wins", "Aerial Win%",
            "Fouls Won"
        ],
        "groups": {
            "Pressure Regains": "Off The Ball",
            "All Goals": "Attacking",
            "Penalty Goals": "Attacking",
            "xG": "Attacking",
            "Shots": "Attacking",
            "xG/Shot": "Attacking",
            "Shot Touch%": "Attacking",
            "Touches In Box": "Attacking",
            "xG Assisted": "Attacking",
            "Aerial Wins": "Off The Ball",
            "Aerial Win%": "Off The Ball",
            "Fouls Won": "Possession",
        }
    },
}

# Colors
group_colors = {
    "Off The Ball": "crimson",
    "Attacking": "royalblue",
    "Possession": "seagreen",
    "Defensive": "darkorange",
    "Goalkeeping": "purple"
}

# ---------- File upload ----------
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if not uploaded_file:
    st.stop()

df = pd.read_excel(uploaded_file)

# ---------- NORMALISE new-provider identifiers ----------
rename_map = {}
if "Name" in df.columns: rename_map["Name"] = "Player"
if "Primary Position" in df.columns: rename_map["Primary Position"] = "Position"
if "Minutes" in df.columns: rename_map["Minutes"] = "Minutes played"
df.rename(columns=rename_map, inplace=True)

# Build "Positions played" using Primary + Secondary
if "Position" in df.columns:
    if "Secondary Position" in df.columns:
        df["Positions played"] = df["Position"].fillna("").astype(str) + np.where(
            df["Secondary Position"].notna() & (df["Secondary Position"].astype(str)!=""),
            ", " + df["Secondary Position"].astype(str),
            ""
        )
    else:
        df["Positions played"] = df["Position"].astype(str)
else:
    df["Positions played"] = np.nan

# Team within selected timeframe fallback
if "Team within selected timeframe" not in df.columns:
    if "Team" in df.columns:
        df["Team within selected timeframe"] = df["Team"]
    else:
        df["Team within selected timeframe"] = np.nan

# Height fallback
if "Height" not in df.columns:
    df["Height"] = np.nan

# Six-Group mapping from PRIMARY position only
if "Position" in df.columns:
    df["Six-Group Position"] = df["Position"].apply(map_first_position_to_group)
else:
    df["Six-Group Position"] = np.nan

# ---------- Duplicate generic CMs into both 6 & 8 ----------
if "Six-Group Position" in df.columns:
    cm_mask = df["Six-Group Position"] == "Centre Midfield"
    if cm_mask.any():
        cm_rows = df.loc[cm_mask].copy()
        cm_as_6 = cm_rows.copy(); cm_as_6["Six-Group Position"] = "Number 6"
        cm_as_8 = cm_rows.copy(); cm_as_8["Six-Group Position"] = "Number 8"
        df = pd.concat([df, cm_as_6, cm_as_8], ignore_index=True)

# ---------- Minutes filter ----------
minutes_col = "Minutes played"
if minutes_col not in df.columns:
    df[minutes_col] = np.nan

min_minutes = st.number_input("Minimum minutes to include", min_value=0, value=1000, step=50)
df["_minutes_numeric"] = pd.to_numeric(df[minutes_col], errors="coerce")
df = df[df["_minutes_numeric"] >= min_minutes].copy()
if df.empty:
    st.warning("No players meet the minutes threshold. Lower the minimum.")
    st.stop()

# ---------- Age slider filter ----------
if "Age" in df.columns:
    df["_age_numeric"] = pd.to_numeric(df["Age"], errors="coerce")
    if df["_age_numeric"].notna().any():
        age_min = int(np.nanmin(df["_age_numeric"]))
        age_max = int(np.nanmax(df["_age_numeric"]))
        sel_min, sel_max = st.slider("Age range to include", min_value=age_min, max_value=age_max, value=(age_min, age_max), step=1)
        df = df[df["_age_numeric"].between(sel_min, sel_max)].copy()
    else:
        st.info("Age column has no numeric values, age filter skipped.")
else:
    st.info("No Age column found, age filter skipped.")

st.caption(f"Filtering on '{minutes_col}' ≥ {min_minutes}. Players remaining, {len(df)}")

# ---------- Group filter ----------
available_groups = [g for g in SIX_GROUPS if "Six-Group Position" in df.columns and g in df["Six-Group Position"].unique()]
selected_groups = st.multiselect("Include groups", options=available_groups, default=[], label_visibility="collapsed")
if selected_groups:
    df = df[df["Six-Group Position"].isin(selected_groups)].copy()
    if df.empty:
        st.warning("No players after group filter. Clear filters or choose different groups.")
        st.stop()

current_single_group = selected_groups[0] if len(selected_groups) == 1 else None

# ---------- Session state ----------
if "selected_player" not in st.session_state:
    st.session_state.selected_player = None
if "selected_template" not in st.session_state:
    st.session_state.selected_template = None
if "last_auto_group" not in st.session_state:
    st.session_state.last_auto_group = None
if "ec_rows" not in st.session_state:
    st.session_state.ec_rows = 1

# Initialise template (prefer single-group default)
if st.session_state.selected_template is None:
    if current_single_group:
        st.session_state.selected_template = DEFAULT_TEMPLATE.get(current_single_group, list(position_metrics.keys())[0])
        st.session_state.last_auto_group = current_single_group
    else:
        st.session_state.selected_template = list(position_metrics.keys())[0]

# Snap to group default if user narrows to exactly one group
if current_single_group is not None and current_single_group != st.session_state.last_auto_group:
    st.session_state.selected_template = DEFAULT_TEMPLATE.get(current_single_group, st.session_state.selected_template)
    st.session_state.last_auto_group = current_single_group

# ---------- Build metric pool for Essential Criteria ----------
current_template_name = st.session_state.selected_template or list(position_metrics.keys())[0]
current_metrics = position_metrics[current_template_name]["metrics"]

for m in current_metrics:
    if m not in df.columns:
        df[m] = 0
df[current_metrics] = df[current_metrics].fillna(0)

# ---------- Essential Criteria ----------
with st.expander("Essential Criteria", expanded=False):
    use_all_cols = st.checkbox("Pick from all numeric columns", value=False)
    numeric_cols_all = sorted([c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])
    metric_pool_base = numeric_cols_all if use_all_cols else current_metrics

    cbtn1, cbtn2, cbtn3 = st.columns(3)
    with cbtn1:
        if st.button("Add criterion"):
            st.session_state.ec_rows += 1
    with cbtn2:
        if st.button("Remove last", disabled=st.session_state.ec_rows <= 1):
            st.session_state.ec_rows = max(1, st.session_state.ec_rows - 1)
    with cbtn3:
        apply_nonneg = st.checkbox("Apply all criteria", value=False)

    if len(metric_pool_base) == 0:
        st.info("No numeric metrics available to filter.")
        apply_nonneg = False
        st.session_state.ec_rows = 1

    criteria = []
    for i in range(st.session_state.ec_rows):
        st.markdown(f"**Criterion {i+1}**")
        c1, c2, c3, c4 = st.columns([3, 2, 2, 3])

        prev_key_metric = f"ec_metric_{i}"
        prev_metric = st.session_state.get(prev_key_metric, None)
        metric_pool_display = list(metric_pool_base)
        if prev_metric and prev_metric not in metric_pool_display and prev_metric in numeric_cols_all:
            metric_pool_display = [prev_metric] + [m for m in metric_pool_display if m != prev_metric]

        with c1:
            metric_name = st.selectbox("Metric", metric_pool_display, key=prev_key_metric)

        with c2:
            mode = st.radio("Apply to", ["Raw", "Percentile"], horizontal=True, key=f"ec_mode_{i}")

        with c3:
            op = st.selectbox("Operator", [">=", ">", "<=", "<"], index=0, key=f"ec_op_{i}")

        with c4:
            if mode == "Percentile":
                default_thr = 50.0
            else:
                default_thr = float(np.nanmedian(pd.to_numeric(df[metric_name], errors="coerce")))
                if not np.isfinite(default_thr):
                    default_thr = 0.0
            thr_str = st.text_input("Threshold", value=str(int(default_thr)), key=f"ec_thr_{i}")
            try:
                thr_val = float(thr_str)
            except ValueError:
                thr_val = default_thr

        criteria.append((metric_name, mode, op, thr_val))

    if apply_nonneg and len(criteria) > 0:
        temp_cols = []
        mask_all = pd.Series(True, index=df.index)

        for metric_name, mode, op, thr_val in criteria:
            if mode == "Percentile":
                df[metric_name] = pd.to_numeric(df[metric_name], errors="coerce")
                perc_series = (df[metric_name].rank(pct=True) * 100).round(1)
                tmp_col = f"__tmp_percentile__{metric_name}"
                df[tmp_col] = perc_series
                filter_col = tmp_col
                temp_cols.append(tmp_col)
            else:
                filter_col = metric_name
                df[filter_col] = pd.to_numeric(df[filter_col], errors="coerce")

            if op == ">=":
                mask = df[filter_col] >= thr_val
            elif op == ">":
                mask = df[filter_col] > thr_val
            elif op == "<=":
                mask = df[filter_col] <= thr_val
            else:
                mask = df[filter_col] < thr_val

            mask_all &= mask

        kept = int(mask_all.sum())
        dropped = int((~mask_all).sum())
        df = df[mask_all].copy()

        if temp_cols:
            df.drop(columns=temp_cols, inplace=True, errors="ignore")

        summary = " AND ".join([f"{m} {o} {t}{'%' if md=='Percentile' else ''}" for m, md, o, t in criteria])
        st.caption(f"Essential Criteria applied: {summary}. Kept {kept}, removed {dropped} players.")

# ---------- Player list (names only; no role suffix) ----------
if "Player" not in df.columns:
    st.error("Expected a 'Name' column in the upload (renamed to 'Player').")
    st.stop()

players = df["Player"].dropna().unique().tolist()
if not players:
    st.warning("No players available after filters.")
    st.stop()

if st.session_state.selected_player not in players:
    st.session_state.selected_player = players[0]

selected_player = st.selectbox(
    "Choose a player",
    players,
    index=players.index(st.session_state.selected_player) if st.session_state.selected_player in players else 0,
    key="player_select"
)
st.session_state.selected_player = selected_player

# ---------- Template select ----------
template_names = list(position_metrics.keys())
tpl_index = template_names.index(st.session_state.selected_template) if st.session_state.selected_template in template_names else 0
selected_position_template = st.selectbox(
    "Choose a position template for the chart",
    template_names,
    index=tpl_index,
    key="template_select"
)
st.session_state.selected_template = selected_position_template

# ---------- Metrics + percentiles ----------
metrics = position_metrics[selected_position_template]["metrics"]
metric_groups = position_metrics[selected_position_template]["groups"]

for m in metrics:
    if m not in df.columns:
        df[m] = 0
df[metrics] = df[metrics].fillna(0)

metrics_df = df[metrics].copy()
percentile_df = (metrics_df.rank(pct=True) * 100).round(1)

keep_cols = ["Player", "Team within selected timeframe", "Team", "Age", "Height", "Positions played", "Minutes played", "Six-Group Position"]
for c in keep_cols:
    if c not in df.columns:
        df[c] = np.nan

plot_data = pd.concat([df[keep_cols], metrics_df, percentile_df.add_suffix(" (percentile)")], axis=1)

sel_metrics = list(metric_groups.keys())
percentiles_all = plot_data[[m + " (percentile)" for m in sel_metrics]]
z_scores_all = (percentiles_all - 50) / 15
plot_data["Avg Z Score"] = z_scores_all.mean(axis=1)
plot_data["Rank"] = plot_data["Avg Z Score"].rank(ascending=False, method="min").astype(int)

# ---------- Chart ----------
def plot_radial_bar_grouped(player_name, plot_data, metric_groups, group_colors):
    row = plot_data[plot_data["Player"] == player_name]
    if row.empty:
        st.error(f"No entry for '{player_name}' found.")
        return

    sel_metrics_loc = list(metric_groups.keys())
    raw = row[sel_metrics_loc].values.flatten()
    percentiles = row[[m + " (percentile)" for m in sel_metrics_loc]].values.flatten()
    groups = [metric_groups[m] for m in sel_metrics_loc]
    colors = [group_colors.get(g, "grey") for g in groups]

    num_bars = len(sel_metrics_loc)
    angles = np.linspace(0, 2*np.pi, num_bars, endpoint=False)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)

    ax.bar(angles, percentiles, width=2*np.pi/num_bars*0.9, color=colors, edgecolor=colors, alpha=0.75)

    # Raw value labels on bars (optional)
    for angle, raw_val in zip(angles, raw):
        try:
            ax.text(angle, 50, f"{float(raw_val):.2f}", ha="center", va="center", color="black", fontsize=10, fontweight="bold")
        except Exception:
            ax.text(angle, 50, "-", ha="center", va="center", color="black", fontsize=10, fontweight="bold")

    # Metric labels outside
    for i, angle in enumerate(angles):
        label = sel_metrics_loc[i]
        ax.text(angle, 108, label, ha="center", va="center", color="black", fontsize=10, fontweight="bold")

    # Group labels
    group_positions = {}
    for g, a in zip(groups, angles):
        group_positions.setdefault(g, []).append(a)
    for group, group_angles in group_positions.items():
        mean_angle = np.mean(group_angles)
        ax.text(mean_angle, 125, group, ha="center", va="center", fontsize=20, fontweight="bold", color=group_colors.get(group, "grey"))

    # Title
    age = row["Age"].values[0]
    height = row["Height"].values[0]
    team = row["Team within selected timeframe"].values[0]
    mins = row["Minutes played"].values[0]
    rank_val = int(row["Rank"].values[0])

    age_str = f"{int(age)} years old" if not pd.isnull(age) else ""
    height_str = f"{int(height)} cm" if not pd.isnull(height) else ""
    parts = [row["Player"].values[0]]
    if age_str: parts.append(age_str)
    if height_str: parts.append(height_str)
    line1 = " | ".join(parts)

    team_str = f"{team}" if pd.notnull(team) else ""
    mins_str = f"{int(mins)} mins" if pd.notnull(mins) else ""
    rank_str = f"Rank #{rank_val}" if pd.notnull(rank_val) else ""
    role_str = row["Six-Group Position"].values[0] if pd.notnull(row["Six-Group Position"].values[0]) else ""
    line2 = " | ".join([p for p in [role_str, team_str, mins_str, rank_str] if p])

    ax.set_title(f"{line1}\n{line2}", color="black", size=22, pad=20, y=1.12)

    # Badge
    z_scores = (percentiles - 50) / 15
    avg_z = float(np.mean(z_scores))

    if avg_z >= 1.0:
        badge = ("Excellent", "#228B22")
    elif avg_z >= 0.3:
        badge = ("Good", "#1E90FF")
    elif avg_z >= -0.3:
        badge = ("Average", "#DAA520")
    else:
        badge = ("Below Average", "#DC143C")

    st.markdown(
        f"<div style='text-align:center; margin-top: 20px;'>"
        f"<span style='font-size:24px; font-weight:bold;'>Average Z Score, {avg_z:.2f}</span><br>"
        f"<span style='background-color:{badge[1]}; color:white; padding:5px 10px; border-radius:8px; font-size:20px;'>{badge[0]}</span></div>",
        unsafe_allow_html=True
    )

    # Club logo in centre
    if logo is not None:
        try:
            img = np.array(logo)
            imagebox = OffsetImage(img, zoom=0.2)
            ab = AnnotationBbox(imagebox, (0, 0), frameon=False, box_alignment=(0.5, 0.5))
            ax.add_artist(ab)
        except Exception as e:
            st.error(f"Could not add logo to chart: {e}")

    st.pyplot(fig, use_container_width=True)

if st.session_state.selected_player:
    plot_radial_bar_grouped(st.session_state.selected_player, plot_data, metric_groups, group_colors)

# ---------- Ranking table ----------
st.markdown("### Players Ranked by Z-Score")
cols_for_table = [
    "Player", "Positions played",
    "Age", "Team", "Team within selected timeframe",
    "Minutes played", "Avg Z Score", "Rank"
]
for c in cols_for_table:
    if c not in plot_data.columns:
        plot_data[c] = np.nan

z_ranking = plot_data[cols_for_table].sort_values(by="Avg Z Score", ascending=False).reset_index(drop=True)
z_ranking[["Team", "Team within selected timeframe"]] = z_ranking[["Team", "Team within selected timeframe"]].fillna("N/A")
if "Age" in z_ranking:
    z_ranking["Age"] = z_ranking["Age"].apply(lambda x: int(x) if pd.notnull(x) else x)
z_ranking.index = np.arange(1, len(z_ranking) + 1)
z_ranking.index.name = "Row"

st.dataframe(z_ranking, use_container_width=True)
