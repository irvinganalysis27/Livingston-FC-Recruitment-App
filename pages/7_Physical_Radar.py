import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import re
from openai import OpenAI

# ========= PAGE CONFIG =========
st.set_page_config(page_title="SkillCorner Physical Radar", layout="centered")
st.title("📈 Physical Radar (SkillCorner)")

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

# ========= BASIC HELPERS =========
def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(u"\xa0", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )
    return df

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

# ========= SKILLCORNER API LOADER =========
try:
    from skillcorner.client import SkillcornerClient

    @st.cache_data(ttl=86400, show_spinner=True)
    def fetch_skillcorner_api():
        client = SkillcornerClient(
            username=st.secrets["SKILLCORNER"]["USERNAME"],
            password=st.secrets["SKILLCORNER"]["PASSWORD"]
        )
        data = client.get_physical(params={
            'competition': '18,459,305',  # Example: Scotland, Championship, etc.
            'season': 2025,
            'group_by': 'player,team,competition,season,group',
            'playing_time__gte': 60,
            'count_match__gte': 5,
            'data_version': '3'
        })
        return pd.DataFrame(data)
except Exception:
    fetch_skillcorner_api = None

# ========= CSV FALLBACK LOADER =========
@st.cache_data(ttl=86400, show_spinner=True)
def load_skillcorner_csv(path_str: str):
    trials = [
        dict(sep=None, engine="python", on_bad_lines="skip"),
        dict(sep=",", engine="python", on_bad_lines="skip"),
        dict(sep="\t", engine="python", on_bad_lines="skip"),
    ]
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    for kwargs in trials:
        try:
            df = pd.read_csv(path, **kwargs)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    raise ValueError("Unable to read SkillCorner CSV")

# ========= LOAD DATA =========
st.caption(f"Source file: `{DATA_PATH}`")
USE_API = st.toggle("Fetch live from SkillCorner API", value=False)
if USE_API and fetch_skillcorner_api:
    df_raw = fetch_skillcorner_api()
else:
    df_raw = load_skillcorner_csv(str(DATA_PATH))

if df_raw.empty:
    st.error("❌ No rows in dataset.")
    st.stop()

df_raw = _clean_columns(df_raw)

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
          .agg({**agg_spec,
                "Team": _last_non_null,
                "Competition": _last_non_null,
                "Season": _last_non_null})
          .reset_index()
)

# ========= FILTERS =========
st.markdown("#### Filters")
leagues = sorted(pd.Series(df_player["Competition"]).dropna().unique().tolist())
selected_leagues = st.multiselect("Leagues", options=leagues, default=leagues, key="sc_league_sel")

pos_groups = sorted(pd.Series(df_player["Position Group Normalised"]).dropna().unique().tolist())
selected_pos_groups = st.multiselect("Position Groups", options=pos_groups, default=pos_groups, key="sc_pos_sel")

min_minutes = st.number_input("Minimum total minutes", min_value=0, value=600, step=60)

df = df_player.copy()
if selected_leagues:
    df = df[df["Competition"].isin(selected_leagues)]
if selected_pos_groups:
    df = df[df["Position Group Normalised"].isin(selected_pos_groups)]
df = df[df["Minutes"] >= min_minutes]

st.caption(f"Players after filters: **{len(df)}**")
if df.empty:
    st.stop()

# ========= PERCENTILES (within league) =========
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

# ========= GLOBAL POSITION SCORES (Z-SCORE BASED) =========
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
    (global_scores.mean(axis=1) - global_scores.mean(axis=1).min()) /
    (global_scores.mean(axis=1).max() - global_scores.mean(axis=1).min()) * 100
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

# ========= AI SUMMARY =========
client = OpenAI(api_key=st.secrets["OpenAI"]["OPENAI_API_KEY"])

def generate_ai_summary(player_name: str):
    try:
        row = df.loc[df["Player"] == player_name].iloc[0]
    except IndexError:
        return "No data available for this player."
    role = str(row.get("Position Group Normalised", "player"))
    league = str(row.get("Competition", ""))
    team = str(row.get("Team", ""))
    mins = row.get("Minutes", 0)
    score = row.get("_score_0_100_global", 0)
    metric_text = ", ".join([f"{m}: {round(row[m],1)}" for m in RADAR_METRICS if pd.notnull(row[m])])

    prompt = f"""
    You are writing a concise, honest scouting summary in the style of Tom Irving,
    focusing on the player's physical profile.

    Write 5–6 sentences about {player_name}, a {role.lower()} in {league} for {team}.
    He has played {mins} minutes and has a physical composite score of {score}/100.
    Use this data: {metric_text}.

    - Highlight strengths (above 70th percentile) and weaknesses (below 40th).
    - Use natural, realistic football analysis language.
    - Finish with one sentence that sums up his physical type or suitability.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            temperature=0.85,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ AI summary generation failed: {e}"

if st.button("🧠 Generate AI Summary"):
    st.markdown("#### AI-Generated Player Summary")
    st.write(generate_ai_summary(selected_player))

# ========= RANKING TABLE =========
st.markdown("### Players Ranked by Physical Composite (0–100)")
rank_cols = ["Player", "Team", "Competition", "Position Group Normalised", "Minutes", "_score_0_100_global"] + RADAR_METRICS
table = df[rank_cols].copy()
table = table.sort_values("_score_0_100_global", ascending=False).reset_index(drop=True)
table.index = np.arange(1, len(table) + 1)
table.index.name = "Rank"
st.dataframe(table, use_container_width=True)
