import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from PIL import Image  # add "Pillow" to requirements.txt if it's not already there

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

logo_path = ASSETS_DIR / "Livingston_FC_club_badge_new.png"
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

# ===================== 7 POSITIONS (NEW) =====================
SEVEN_GROUPS = [
    "Goalkeeper",
    "Centre Back",
    "Full Back",
    "Number 6",
    "Number 8",
    "Winger",
    "Striker",
]

def _clean_pos_token(tok: str) -> str:
    if pd.isna(tok):
        return ""
    t = str(tok).upper()
    t = t.replace(".", "").replace("-", "").replace(" ", "")
    return t

def parse_first_position(cell) -> str:
    if pd.isna(cell):
        return ""
    first = re.split(r"[,/]", str(cell))[0].strip()
    return _clean_pos_token(first)

# Map raw position tokens into your 7 buckets
RAW_TO_SEVEN = {
    # Goalkeeper
    "GK":"Goalkeeper", "GKP":"Goalkeeper", "GOALKEEPER":"Goalkeeper",

    # Centre Back
    "CB":"Centre Back", "RCB":"Centre Back", "LCB":"Centre Back",
    "CBR":"Centre Back", "CBL":"Centre Back", "SW":"Centre Back", "CDF":"Centre Back",

    # Full Back
    "RB":"Full Back", "LB":"Full Back", "RWB":"Full Back", "LWB":"Full Back",
    "RFB":"Full Back", "LFB":"Full Back", "WB":"Full Back",

    # Number 6 (defensive CM)
    "DMF":"Number 6", "DM":"Number 6", "CDM":"Number 6",
    "RDMF":"Number 6", "LDMF":"Number 6",

    # Number 8 (all-round CM / CM/AM hybrid)
    "CMF":"Number 8", "CM":"Number 8", "RCMF":"Number 8", "LCMF":"Number 8",
    "AMF":"Number 8", "AM":"Number 8", "CAM":"Number 8", "SS":"Number 8", "10":"Number 8",

    # Winger
    "RW":"Winger", "LW":"Winger", "RWF":"Winger", "LWF":"Winger",
    "RM":"Winger", "LM":"Winger", "WF":"Winger", "W":"Winger",

    # Striker
    "CF":"Striker", "ST":"Striker", "9":"Striker", "FW":"Striker", "STK":"Striker", "CFW":"Striker"
}

def map_first_position_to_group(cell) -> str:
    tok = parse_first_position(cell)
    return RAW_TO_SEVEN.get(tok, "Winger")  # safe-ish default wide

# ===================== TEMPLATES (7 ONLY) =====================
# These are the *display* metric labels we want to plot for each position.
# A resolver below will match them to real CSV columns via regex synonyms.
position_templates = {
    "Goalkeeper": [
        "Save%", "Goals Conceded", "Claims%", "GK Aggressive Dist.", "Goalkeeper OBV",
        "Pass OBV", "Long Balls", "LBP/Pass%",  # distribution
    ],
    "Centre Back": [
        # Defensive
        "Defensive Duels", "Defensive Duels Won%", "Aerial Duels", "Aerial Duels Won%",
        "Blocks", "Interceptions", "Clearances",
        # Possession
        "Accurate Passes%", "Forward Passes", "Accurate Forward Passes%",
        "Passes to Final Third", "Accurate Passes to Final Third%",
        "Dribbles", "Successful Dribbles%",
    ],
    "Full Back": [
        # Defensive
        "Successful Defensive Actions", "Defensive Duels", "Defensive Duels Won%", "Interceptions",
        # Possession / crossing threat
        "Crosses", "Accurate Crosses%", "Passes to Final Third", "Accurate Passes to Final Third%",
        "Dribbles", "Successful Dribbles%",
        # Attacking
        "xG Assisted", "Assists", "Shot Assists"
    ],
    "Number 6": [
        # Defensive screen
        "Successful Defensive Actions", "Defensive Duels", "Defensive Duels Won%",
        "Aerial Duels", "Aerial Duels Won%", "Interceptions",
        # Build-up
        "Accurate Passes%", "Forward Passes", "Accurate Forward Passes%",
        "Passes to Final Third", "Accurate Passes to Final Third%",
        "Pass OBV"
    ],
    "Number 8": [
        # Attacking & box arrivals
        "NP Goals", "xG", "Shots", "Shots on Target%", "Goal Conversion%",
        "Assists", "xG Assisted",
        # Progression
        "Forward Passes", "Accurate Forward Passes%", "Passes to Final Third", "Accurate Passes to Final Third%",
        # Carry & dribble
        "Dribbles", "Successful Dribbles%"
    ],
    "Winger": [
        "NP Goals", "xG", "Assists", "xG Assisted",
        "Crosses", "Accurate Crosses%", "Dribbles", "Successful Dribbles%",
        "Fouls Suffered", "Shot Assists",
        "Passes to Penalty Area", "Accurate Passes to Penalty Area%",
        "Touches In Box"
    ],
    "Striker": [
        "Successful Defensive Actions", "Aerial Duels", "Aerial Duels Won%",
        "NP Goals", "xG", "Shots", "Shots on Target%", "Goal Conversion%",
        "Assists", "xG Assisted", "Shot Assists",
        "Touches In Box"
    ],
}

# Group colors (kept)
group_colors = {
    "Off The Ball": "crimson",
    "Attacking": "royalblue",
    "Possession": "seagreen",
    "Defensive": "darkorange",
    "Goalkeeping": "purple"
}

# Metric groups per template (display labels → group)
position_groups = {
    "Goalkeeper": {
        "Save%":"Goalkeeping", "Goals Conceded":"Goalkeeping", "Claims%":"Goalkeeping",
        "GK Aggressive Dist.":"Goalkeeping", "Goalkeeper OBV":"Goalkeeping",
        "Pass OBV":"Possession", "Long Balls":"Possession", "LBP/Pass%":"Possession",
    },
    "Centre Back": {
        "Defensive Duels":"Defensive", "Defensive Duels Won%":"Defensive",
        "Aerial Duels":"Defensive", "Aerial Duels Won%":"Defensive",
        "Blocks":"Defensive", "Interceptions":"Defensive", "Clearances":"Defensive",
        "Accurate Passes%":"Possession", "Forward Passes":"Possession", "Accurate Forward Passes%":"Possession",
        "Passes to Final Third":"Possession", "Accurate Passes to Final Third%":"Possession",
        "Dribbles":"Possession", "Successful Dribbles%":"Possession"
    },
    "Full Back": {
        "Successful Defensive Actions":"Defensive", "Defensive Duels":"Defensive",
        "Defensive Duels Won%":"Defensive", "Interceptions":"Defensive",
        "Crosses":"Possession", "Accurate Crosses%":"Possession",
        "Passes to Final Third":"Possession", "Accurate Passes to Final Third%":"Possession",
        "Dribbles":"Possession", "Successful Dribbles%":"Possession",
        "xG Assisted":"Attacking", "Assists":"Attacking", "Shot Assists":"Attacking"
    },
    "Number 6": {
        "Successful Defensive Actions":"Defensive", "Defensive Duels":"Defensive",
        "Defensive Duels Won%":"Defensive", "Aerial Duels":"Defensive",
        "Aerial Duels Won%":"Defensive", "Interceptions":"Defensive",
        "Accurate Passes%":"Possession", "Forward Passes":"Possession",
        "Accurate Forward Passes%":"Possession", "Passes to Final Third":"Possession",
        "Accurate Passes to Final Third%":"Possession", "Pass OBV":"Possession"
    },
    "Number 8": {
        "NP Goals":"Attacking", "xG":"Attacking", "Shots":"Attacking",
        "Shots on Target%":"Attacking", "Goal Conversion%":"Attacking",
        "Assists":"Attacking", "xG Assisted":"Attacking",
        "Forward Passes":"Possession", "Accurate Forward Passes%":"Possession",
        "Passes to Final Third":"Possession", "Accurate Passes to Final Third%":"Possession",
        "Dribbles":"Possession", "Successful Dribbles%":"Possession"
    },
    "Winger": {
        "NP Goals":"Attacking", "xG":"Attacking", "Assists":"Attacking", "xG Assisted":"Attacking",
        "Crosses":"Possession", "Accurate Crosses%":"Possession",
        "Dribbles":"Possession", "Successful Dribbles%":"Possession",
        "Fouls Suffered":"Possession", "Shot Assists":"Attacking",
        "Passes to Penalty Area":"Possession", "Accurate Passes to Penalty Area%":"Possession",
        "Touches In Box":"Attacking"
    },
    "Striker": {
        "Successful Defensive Actions":"Off The Ball", "Aerial Duels":"Off The Ball", "Aerial Duels Won%":"Off The Ball",
        "NP Goals":"Attacking", "xG":"Attacking", "Shots":"Attacking", "Shots on Target%":"Attacking",
        "Goal Conversion%":"Attacking", "Assists":"Attacking", "xG Assisted":"Attacking", "Shot Assists":"Attacking",
        "Touches In Box":"Attacking"
    },
}

# Default template per bucket (same-name now)
DEFAULT_TEMPLATE = {g: g for g in SEVEN_GROUPS}

# ===================== METRIC SYNONYMS (resolver) =====================
# Left side: the *display* label we use in charts.
# Right side: list of regex patterns it should match in your CSV.
METRIC_SYNONYMS = {
    # GK
    "Save%": [r"^Save%$"],
    "Goals Conceded": [r"Goals Conceded"],
    "Claims%": [r"Claims%"],
    "GK Aggressive Dist.": [r"GK Aggressive Dist\.?"],
    "Goalkeeper OBV": [r"Goalkeeper OBV"],
    "Pass OBV": [r"Pass OBV"],
    "Long Balls": [r"Long Balls", r"Long Pass(es)?"],
    "LBP/Pass%": [r"LBP/Pass%"],

    # Def actions
    "Defensive Duels": [r"Defensive Duels(?! Won)"],
    "Defensive Duels Won%": [r"Defensive Duels Won.*%"],
    "Aerial Duels": [r"Aerial Duels(?! Won)"],
    "Aerial Duels Won%": [r"Aerial Duels Won.*%"],
    "Blocks": [r"Blocks"],
    "Interceptions": [r"Interceptions|PAdj Interceptions"],
    "Clearances": [r"Clearances"],
    "Successful Defensive Actions": [r"Successful Defensive Actions"],

    # Possession / Passing
    "Accurate Passes%": [r"Accurate Pass(es)?\,?\s*%|Pass Accuracy%"],
    "Forward Passes": [r"Forward Pass(es)?"],
    "Accurate Forward Passes%": [r"Accurate Forward Pass(es)?.*%"],
    "Passes to Final Third": [r"Pass(es)? to Final Third"],
    "Accurate Passes to Final Third%": [r"Accurate Pass(es)? to Final Third.*%"],
    "Passes to Penalty Area": [r"Pass(es)? to Penalty Area"],
    "Accurate Passes to Penalty Area%": [r"Accurate Pass(es)? to Penalty Area.*%"],

    # Carry/Dribble/Cross
    "Dribbles": [r"Dribbles(?!.*%)"],
    "Successful Dribbles%": [r"Successful Dribbles.*%"],
    "Crosses": [r"Cross(es)?(?!.*Acc)"],
    "Accurate Crosses%": [r"(Accurate )?Cross.*%"],

    # Attacking
    "NP Goals": [r"NP Goals"],
    "xG": [r"^xG$"],
    "Shots": [r"^Shots$"],
    "Shots on Target%": [r"Shots on target.*%|SOT%"],
    "Goal Conversion%": [r"Goal Conversion%"],
    "Assists": [r"^Assists$"],
    "xG Assisted": [r"xG Assisted|xA"],
    "Shot Assists": [r"Shot Assists|Key Pass(es)?"],
    "Fouls Suffered": [r"Fouls Suffered"],
    "Touches In Box": [r"Touches In Box"],
}

def resolve_metric_columns(df_cols, desired_labels):
    """
    Returns:
      rename_map: dict of {actual_col_name: display_label}
      missing: list of display_labels we couldn't find
    """
    rename_map = {}
    missing = []
    for disp in desired_labels:
        patterns = METRIC_SYNONYMS.get(disp, [])
        found = None
        for pat in patterns:
            rx = re.compile(pat, re.I)
            for c in df_cols:
                if rx.search(str(c)):
                    found = c
                    break
            if found:
                break
        if found:
            rename_map[found] = disp
        else:
            missing.append(disp)
    return rename_map, missing

# ---------- File upload ----------
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])
if not uploaded_file:
    st.stop()

# Load CSV (preferred) or Excel
if uploaded_file.name.lower().endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

# ---------- Position labelling ----------
pos_col_name = "Position" if "Position" in df.columns else None
if pos_col_name:
    df["Positions played"] = df[pos_col_name].astype(str)
    df["Seven-Group Position"] = df[pos_col_name].apply(map_first_position_to_group)
else:
    df["Positions played"] = np.nan
    df["Seven-Group Position"] = np.nan

# ---------- Minutes filter ----------
minutes_col = "Minutes played" if "Minutes played" in df.columns else None
min_minutes = st.number_input("Minimum minutes to include", min_value=0, value=1000, step=50)
if minutes_col:
    df["_minutes_numeric"] = pd.to_numeric(df[minutes_col], errors="coerce")
    df = df[df["_minutes_numeric"] >= min_minutes].copy()
else:
    st.info("No 'Minutes played' column found; minutes filter skipped.")

if df.empty:
    st.warning("No players meet the minutes threshold. Lower the minimum or check your data file.")
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

if minutes_col:
    st.caption(f"Filtering on '{minutes_col}' ≥ {min_minutes}. Players remaining, {len(df)}")
else:
    st.caption(f"Players remaining, {len(df)}")

# ---------- 7-group filter ----------
if "Seven-Group Position" in df.columns:
    available_groups = [g for g in SEVEN_GROUPS if g in df["Seven-Group Position"].dropna().unique()]
else:
    available_groups = []

selected_groups = st.multiselect("Include groups", options=available_groups, default=[], label_visibility="collapsed")
if selected_groups:
    df = df[df["Seven-Group Position"].isin(selected_groups)].copy()
    if df.empty:
        st.warning("No players after position filter. Clear filters or pick different groups.")
        st.stop()

# Track if exactly one group is selected
current_single_group = selected_groups[0] if len(selected_groups) == 1 else None

# ---------- Session state for player/template & EC ----------
if "selected_player" not in st.session_state:
    st.session_state.selected_player = None
if "selected_template" not in st.session_state:
    st.session_state.selected_template = None
if "last_auto_group" not in st.session_state:
    st.session_state.last_auto_group = None
if "ec_rows" not in st.session_state:
    st.session_state.ec_rows = 1

# Initialise template once (prefer single-group default)
if st.session_state.selected_template is None:
    if current_single_group:
        st.session_state.selected_template = DEFAULT_TEMPLATE.get(current_single_group, list(position_templates.keys())[0])
        st.session_state.last_auto_group = current_single_group
    else:
        st.session_state.selected_template = list(position_templates.keys())[0]

# If the 7-group selection changed to a *new* single group, snap to that default
if current_single_group is not None and current_single_group != st.session_state.last_auto_group:
    st.session_state.selected_template = DEFAULT_TEMPLATE.get(current_single_group, st.session_state.selected_template)
    st.session_state.last_auto_group = current_single_group

# ---------- Template select ----------
template_names = list(position_templates.keys())
tpl_index = template_names.index(st.session_state.selected_template) if st.session_state.selected_template in template_names else 0
selected_position_template = st.selectbox(
    "Choose a position template for the chart",
    template_names,
    index=tpl_index,
    key="template_select"
)
st.session_state.selected_template = selected_position_template

# ---------- Resolve metrics from new provider ----------
desired_metrics = position_templates[selected_position_template]
rename_map, missing_metrics = resolve_metric_columns(df.columns, desired_metrics)

if missing_metrics:
    st.warning(
        "Some metrics weren’t found in your file and will be filled with 0s: " +
        ", ".join(missing_metrics)
    )

# Make a working copy with resolved column names renamed to the *display* labels
df_work = df.copy()
df_work = df_work.rename(columns=rename_map)

# Ensure all desired metric columns exist
for disp in desired_metrics:
    if disp not in df_work.columns:
        df_work[disp] = 0

# ---------- Metric groups for the chosen template ----------
metric_groups = position_groups[selected_position_template]
sel_metrics = list(metric_groups.keys())

# ---------- Percentiles & base columns ----------
metrics_df = df_work[sel_metrics].copy()
percentile_df = (metrics_df.rank(pct=True) * 100).round(1)

keep_cols = ["Player", "Team within selected timeframe", "Team", "Age", "Height", "Positions played", "Minutes played"]
for c in keep_cols:
    if c not in df_work.columns:
        df_work[c] = np.nan

plot_data = pd.concat([df_work[keep_cols], metrics_df, percentile_df.add_suffix(" (percentile)")], axis=1)

percentiles_all = plot_data[[m + " (percentile)" for m in sel_metrics]]
z_scores_all = (percentiles_all - 50) / 15
plot_data["Avg Z Score"] = z_scores_all.mean(axis=1)
plot_data["Rank"] = plot_data["Avg Z Score"].rank(ascending=False, method="min").astype(int)

# ---------- Player select ----------
if "Player" not in df_work.columns:
    st.error("No 'Player' column found in data.")
    st.stop()

players = df_work["Player"].dropna().astype(str).unique().tolist()
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

# ---------- Chart ----------
def plot_radial_bar_grouped(player_name, plot_data, metric_groups, group_colors):
    row = plot_data[plot_data["Player"] == player_name]
    if row.empty:
        st.error(f"No player named '{player_name}' found.")
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

    ax.bar(angles, percentiles, width=2*np.pi/num_bars*0.9, edgecolor=None, alpha=0.75, color=colors)

    for angle, raw_val in zip(angles, raw):
        try:
            ax.text(angle, 50, f"{float(raw_val):.2f}", ha="center", va="center", color="black", fontsize=10, fontweight="bold")
        except Exception:
            ax.text(angle, 50, "-", ha="center", va="center", color="black", fontsize=10, fontweight="bold")

    for i, angle in enumerate(angles):
        label = sel_metrics_loc[i]
        label = label.replace(" per 90", "").replace(", %", " (%)")
        ax.text(angle, 108, label, ha="center", va="center", color="black", fontsize=10, fontweight="bold")

    group_positions = {}
    for g, a in zip(groups, angles):
        group_positions.setdefault(g, []).append(a)
    for group, group_angles in group_positions.items():
        mean_angle = np.mean(group_angles)
        ax.text(mean_angle, 125, group, ha="center", va="center", fontsize=20, fontweight="bold", color=group_colors.get(group, "grey"))

    age = row["Age"].values[0] if "Age" in row else np.nan
    height = row["Height"].values[0] if "Height" in row else np.nan
    team = row["Team within selected timeframe"].values[0] if "Team within selected timeframe" in row else np.nan
    mins = row["Minutes played"].values[0] if "Minutes played" in row else np.nan
    rank_val = int(row["Rank"].values[0]) if "Rank" in row else None

    age_str = f"{int(age)} years old" if pd.notnull(age) else ""
    height_str = f"{int(height)} cm" if pd.notnull(height) else ""
    parts = [player_name]
    if age_str: parts.append(age_str)
    if height_str: parts.append(height_str)
    line1 = " | ".join(parts)

    team_str = f"{team}" if pd.notnull(team) else ""
    mins_str = f"{int(mins)} mins" if pd.notnull(mins) else ""
    rank_str = f"Rank #{rank_val}" if rank_val is not None else ""
    line2 = " | ".join([p for p in [team_str, mins_str, rank_str] if p])

    ax.set_title(f"{line1}\n{line2}", color="black", size=22, pad=20, y=1.12)

    z_scores = (percentiles - 50) / 15
    avg_z = np.mean(z_scores)

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

    st.pyplot(fig, use_container_width=True)

if st.session_state.selected_player:
    plot_radial_bar_grouped(st.session_state.selected_player, plot_data, metric_groups, group_colors)

# ---------- Ranking table ----------
st.markdown("### Players Ranked by Z-Score")
cols_for_table = ["Player", "Positions played", "Age", "Team", "Team within selected timeframe", "Minutes played", "Avg Z Score", "Rank"]
for c in cols_for_table:
    if c not in plot_data.columns:
        plot_data[c] = np.nan
z_ranking = (plot_data[cols_for_table].sort_values(by="Avg Z Score", ascending=False).reset_index(drop=True))
z_ranking[["Team", "Team within selected timeframe"]] = z_ranking[["Team", "Team within selected timeframe"]].fillna("N/A")
if "Age" in z_ranking:
    z_ranking["Age"] = z_ranking["Age"].apply(lambda x: int(x) if pd.notnull(x) else x)
z_ranking.index = np.arange(1, len(z_ranking) + 1)
z_ranking.index.name = "Row"

st.dataframe(z_ranking, use_container_width=True)
