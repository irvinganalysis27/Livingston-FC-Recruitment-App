import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import re
from openai import OpenAI
from auth import check_password
from branding import show_branding

st.set_page_config(page_title="Livingston FC Recruitment App", layout="centered")

# ---------- Authentication ----------
if not check_password():
    st.stop()

# ---------- Branding ----------
show_branding()
st.title("SkillCorner Radar")

# ========= PATHS / CONFIG =========
ROOT_DIR = Path(__file__).parents[1]
DATA_PATH = ROOT_DIR / "SkillCorner-2025-10-18.csv"

# ========= RADAR METRICS =========
RADAR_METRICS = [
    "Distance P90", "M/min P90", "Running Distance P90",
    "HSR Distance P90", "Sprint Count P90",
    "High Acceleration Count P90", "High Deceleration Count P90",
    "PSV-99", "HI Distance P90",
]

METRIC_GROUPS = {
    "Distance P90": "Work Rate", "M/min P90": "Work Rate",
    "Running Distance P90": "Running Load", "HSR Distance P90": "Running Load",
    "Sprint Count P90": "Explosiveness", "High Acceleration Count P90": "Explosiveness",
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

GROUP_THRESHOLDS = dict(high=70.0, low=40.0)

# ========= HELPER FUNCTIONS =========
def compute_group_labels(percentile_row: pd.Series) -> dict:
    """Generate High/Average/Low labels per physical group."""
    groups = {}
    by_group = {}
    for m, g in METRIC_GROUPS.items():
        if m in percentile_row.index:
            by_group.setdefault(g, []).append(pd.to_numeric(percentile_row[m], errors="coerce"))
    for g, vals in by_group.items():
        vals = pd.Series(vals, dtype="float").dropna()
        if vals.empty:
            groups[g] = dict(score=np.nan, label="Unknown")
            continue
        score = float(vals.mean())
        if score >= GROUP_THRESHOLDS["high"]:
            label = "High"
        elif score < GROUP_THRESHOLDS["low"]:
            label = "Low"
        else:
            label = "Average"
        groups[g] = dict(score=round(score, 1), label=label)
    return groups


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names by removing extra spaces and weird characters."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(u"\xa0", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )
    return df


def pct_rank(series: pd.Series, lower_is_better: bool = False) -> pd.Series:
    """Return percentile ranks (0–100)."""
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    r = s.rank(pct=True, ascending=True)
    if lower_is_better:
        r = 1.0 - r
    return (r * 100.0).round(1)


@st.cache_data(ttl=86400, show_spinner=True)
def load_skillcorner_csv(path: Path) -> pd.DataFrame:
    """Load SkillCorner CSV file with fallback parsing."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    for sep in [",", "\t", None]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", on_bad_lines="skip")
            if df.shape[1] > 1:
                return _clean_columns(df)
        except Exception:
            continue
    raise ValueError(f"Unable to read CSV file: {path}")


# ========= LOAD DATA =========
try:
    df_raw = load_skillcorner_csv(DATA_PATH)
except Exception as e:
    st.error(f"❌ Could not load SkillCorner CSV: {e}")
    st.stop()

if df_raw.empty:
    st.error("❌ No rows in dataset.")
    st.stop()

# ========= NORMALISE POSITIONS =========
SKILLCORNER_TO_FIVE = {
    "GK": "Goalkeeper",
    "CB": "Centre Back", "RCB": "Centre Back", "LCB": "Centre Back",
    "RB": "Full Back", "RWB": "Full Back", "LB": "Full Back", "LWB": "Full Back",
    "CDM": "Midfield", "DM": "Midfield", "CM": "Midfield", "RCM": "Midfield",
    "LCM": "Midfield", "CAM": "Midfield", "AM": "Midfield",
    "RW": "Winger", "LW": "Winger", "RM": "Winger", "LM": "Winger",
    "CF": "Striker", "ST": "Striker", "LS": "Striker", "RS": "Striker",
}
df_raw["Position Group Normalised"] = df_raw["Position"].map(SKILLCORNER_TO_FIVE)

# ========= AGGREGATE TO PLAYER LEVEL =========
agg_spec = {m: "mean" for m in RADAR_METRICS}
agg_spec["PSV-99"] = "max"
agg_spec["Minutes"] = "sum"
id_cols = ["Player", "Team", "Competition", "Season", "Position Group Normalised"]

df_sorted = df_raw.sort_values(["Player", "Date"], ascending=[True, True]).copy()
use_cols = list(set(id_cols + ["Date", "Minutes"] + RADAR_METRICS + ["PSV-99"]))
use_cols = [c for c in use_cols if c in df_sorted.columns]
df_use = df_sorted[use_cols].copy()


def _last_non_null(s: pd.Series):
    v = s.dropna()
    return v.iloc[-1] if len(v) else np.nan


df_player = (
    df_use.groupby(["Player", "Position Group Normalised"], dropna=False)
    .agg(
        {
            **agg_spec,
            "Team": _last_non_null,
            "Competition": _last_non_null,
            "Season": _last_non_null,
        }
    )
    .reset_index()
)

# ========= FILTERS =========
st.markdown("#### Filters")

# --- League Filter with Select/Clear All Buttons ---
leagues = sorted(pd.Series(df_player["Competition"]).dropna().unique().tolist())

if "sc_league_sel" not in st.session_state:
    st.session_state.sc_league_sel = leagues

# --- League Filter with Select/Clear All Buttons ---
st.markdown("#### Choose league(s)")

if "sc_league_sel" not in st.session_state:
    st.session_state.sc_league_sel = leagues.copy()

b1, b2, _ = st.columns([1, 1, 6])
with b1:
    if st.button("Select all"):
        st.session_state.sc_league_sel = leagues.copy()
with b2:
    if st.button("Clear all"):
        st.session_state.sc_league_sel = []

valid_defaults = [l for l in st.session_state.get("sc_league_sel", []) if l in leagues]

selected_leagues = st.multiselect(
    "Leagues",
    options=leagues,
    default=valid_defaults,
    key="sc_league_sel",
    label_visibility="collapsed"
)

# Keep session_state clean if options changed
if set(valid_defaults) != set(st.session_state.sc_league_sel):
    st.session_state.sc_league_sel = valid_defaults

# --- Position Filter (Centre Back default) ---
pos_groups = sorted(
    pd.Series(df_player["Position Group Normalised"]).dropna().unique().tolist()
)
default_pos_groups = ["Centre Back"] if "Centre Back" in pos_groups else []
selected_pos_groups = st.multiselect(
    "Position Groups",
    options=pos_groups,
    default=default_pos_groups,
    key="sc_pos_sel",
)

# --- Minutes Filter ---
min_minutes = st.number_input("Minimum total minutes", min_value=0, value=600, step=60)

# --- Apply Filters ---
df = df_player.copy()
if selected_leagues:
    df = df[df["Competition"].isin(selected_leagues)]
if selected_pos_groups:
    df = df[df["Position Group Normalised"].isin(selected_pos_groups)]
df = df[df["Minutes"] >= min_minutes]

st.caption(f"Players after filters: **{len(df)}**")
if df.empty:
    st.stop()

# ========= PERCENTILES =========
compute_within_league = st.checkbox("Percentiles within each league", value=True)
percentile_df = pd.DataFrame(index=df.index, columns=RADAR_METRICS, dtype=float)

if compute_within_league and "Competition" in df.columns:
    for m in RADAR_METRICS:
        percentile_df[m] = df.groupby("Competition", group_keys=False)[m].apply(lambda s: pct_rank(s))
else:
    for m in RADAR_METRICS:
        percentile_df[m] = pct_rank(df[m])

percentile_df = percentile_df.fillna(50.0).round(1)
df["_score_0_100_league"] = percentile_df.mean(axis=1).round(1)

# ========= GLOBAL Z-SCORE =========
global_scores = pd.DataFrame(index=df.index, columns=RADAR_METRICS, dtype=float)
for m in RADAR_METRICS:
    try:
        global_scores[m] = (
            df.groupby("Position Group Normalised", group_keys=False)[m]
            .transform(lambda s: (s - s.mean()) / s.std(ddof=0))
        )
    except Exception:
        global_scores[m] = 0.0

df["_score_0_100_global"] = (
    (global_scores.mean(axis=1) - global_scores.mean(axis=1).min())
    / (global_scores.mean(axis=1).max() - global_scores.mean(axis=1).min())
    * 100
).round(1)

# ========= PLAYER SELECT =========
players = sorted(df["Player"].dropna().unique().tolist())
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
        st.error(f"No player named '{player_name}'")
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

    for ang, raw_val in zip(angles, raws):
        txt = f"{raw_val:.1f}" if np.isfinite(raw_val) else "-"
        ax.text(ang, 50, txt, ha="center", va="center", color="black", fontsize=10, fontweight="bold")

    for ang, m in zip(angles, RADAR_METRICS):
        group = METRIC_GROUPS.get(m, "Other")
        color = GROUP_COLOURS.get(group, "black")
        ax.text(ang, 108, m, ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=GROUP_COLOURS.get(g, "grey"), label=g)
               for g in list(dict.fromkeys(METRIC_GROUPS.values()))]
    ax.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=4, frameon=False)

    team = str(row.get("Team") or "")
    comp = str(row.get("Competition") or "")
    mins = row.get("Minutes")
    posg = str(row.get("Position Group Normalised") or "")
    line1 = " | ".join([x for x in [player_name, posg] if x])
    line2 = " | ".join([x for x in [team, comp, f"{int(mins)} mins", f"{float(row.get('_score_0_100_global', np.nan)):.0f}/100"] if x])
    ax.set_title(f"{line1}\n{line2}", color="black", size=22, pad=20, y=1.10)
    st.pyplot(fig, use_container_width=True)


plot_radial_bar_grouped(selected_player)

# ========= AI SCOUTING SUMMARY =========
st.markdown("### 🧠 AI Scouting Summary")

# --- Setup OpenAI client ---
client = OpenAI(api_key=st.secrets["OpenAI"]["OPENAI_API_KEY"])

def generate_ai_summary(player_name: str, df: pd.DataFrame, percentiles: pd.DataFrame):
    """Generate a short professional scouting summary for the selected player."""
    try:
        row = df.loc[df["Player"] == player_name].iloc[0]
    except IndexError:
        return "No data found for this player."

    # Collect player info
    position = str(row.get("Position Group Normalised", ""))
    league = str(row.get("Competition", ""))
    team = str(row.get("Team", ""))
    mins = int(row.get("Minutes", 0))

    # Get percentile values for metrics
    pcts = percentiles.loc[df["Player"] == player_name, RADAR_METRICS].iloc[0].to_dict()
    pct_text = ", ".join([f"{m}: {v:.0f}" for m, v in pcts.items() if pd.notnull(v)])

    # Build natural-language prompt
    prompt = f"""
    You are a professional football recruitment analyst writing a concise scouting report 
    in the realistic, honest tone of Tom Irving — analytical, grounded, and professional.

    Write 5–6 sentences on {player_name}, a {position.lower()} playing for {team} in {league}, 
    with {mins} minutes of tracked physical data.

    Use these physical percentile metrics (0–100 scale):
    {pct_text}

    Tone and writing style guidelines:
    - Use realistic football language, not clichés.
    - Highlight clear physical traits like running load, explosiveness, or top speed.
    - If the player ranks high (≥70th percentile) in multiple areas, describe them as strong or standout physically.
    - If the player ranks low (<40th), note limitations naturally (e.g. “less active off the ball”, “lacks repeat sprints”).
    - Keep the tone factual and balanced, not promotional.
    - End with one strong sentence summarising their athletic profile type (e.g. “Profiles as an intense, high-output runner suited to pressing systems.”).
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.85,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ AI summary generation failed: {e}"

# --- Button and output ---
if st.button("Generate AI Summary", key="ai_summary_btn"):
    with st.spinner("Generating AI scouting report..."):
        ai_report = generate_ai_summary(selected_player, df, percentile_df)
        st.markdown(ai_report)

# ========= RANKING TABLE =========
st.markdown("### Players Ranked by Physical Composite (0–100)")
rank_cols = ["Player", "Team", "Competition", "Position Group Normalised", "Minutes", "_score_0_100_global"] + RADAR_METRICS
table = df[rank_cols].copy()
table = table.sort_values("_score_0_100_global", ascending=False).reset_index(drop=True)
table.index = np.arange(1, len(table) + 1)
table.index.name = "Rank"
table.rename(columns={"_score_0_100_global": "Score (0–100)"}, inplace=True)
st.dataframe(table, use_container_width=True)
