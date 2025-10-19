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

# ---- Add after GROUP_COLOURS ----
GROUP_THRESHOLDS = dict(high=70.0, low=40.0)

# The metric → group mapping already exists as METRIC_GROUPS.
# We'll compute one score per group = mean of its metric percentiles.
def compute_group_labels(percentile_row: pd.Series) -> dict:
    # percentile_row is a single player's percentiles (the row from `percentile_df`)
    groups = {}
    # bucket metric percentiles by group, then mean
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

# ========= BASIC HELPERS =========
def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names by removing extra spaces and non-breaking spaces."""
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

# ========= CSV LOADER =========
@st.cache_data(ttl=86400, show_spinner=True)
def load_skillcorner_csv(path: Path) -> pd.DataFrame:
    """Load SkillCorner CSV file with basic safety and fallback parsing."""
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
st.caption(f"Source file: `{DATA_PATH}`")

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
          .agg({**agg_spec,
                "Team": _last_non_null,
                "Competition": _last_non_null,
                "Season": _last_non_null})
          .reset_index()
)

# ========= FILTERS =========
st.markdown("#### Filters")

# --- League Filter with Select/Clear All Buttons ---
leagues = sorted(pd.Series(df_player["Competition"]).dropna().unique().tolist())

# Initialise session state for league selection
if "sc_league_sel" not in st.session_state:
    st.session_state.sc_league_sel = leagues

# Buttons for quick selection
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("✅ Add All Leagues"):
        st.session_state.sc_league_sel = leagues
with col2:
    if st.button("❌ Remove All Leagues"):
        st.session_state.sc_league_sel = []

# Multi-select widget
selected_leagues = st.multiselect(
    "Leagues",
    options=leagues,
    default=st.session_state.sc_league_sel,
    key="sc_league_sel"
)

# --- Position Filter (Centre Back only by default) ---
pos_groups = sorted(pd.Series(df_player["Position Group Normalised"]).dropna().unique().tolist())

# Set default to only "Centre Back"
default_pos_groups = ["Centre Back"] if "Centre Back" in pos_groups else []
selected_pos_groups = st.multiselect(
    "Position Groups",
    options=pos_groups,
    default=default_pos_groups,
    key="sc_pos_sel"
)

# --- Minimum Minutes Input ---
min_minutes = st.number_input("Minimum total minutes", min_value=0, value=600, step=60)

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

# ---- Replace your current summary function with this one ----
from openai import OpenAI
client = OpenAI(api_key=st.secrets["OpenAI"]["OPENAI_API_KEY"])

def generate_ai_summary(player_name: str, df_players: pd.DataFrame, percentile_df: pd.DataFrame) -> str:
    # Find the player's row
    row_df = df_players.loc[df_players["Player"] == player_name]
    if row_df.empty:
        return "No data available for this player."
    row = row_df.iloc[0]

    # Pull that player’s percentiles row
    p_row = percentile_df.loc[row_df.index[0], RADAR_METRICS]
    group_info = compute_group_labels(p_row)

    # Build a locked “evidence” block the model must follow
    evidence_lines = []
    for g in ["Work Rate", "Running Load", "Explosiveness", "Top Speed", "Intensity"]:
        if g in group_info:
            gi = group_info[g]
            evidence_lines.append(f"- {g}: {gi['label']} ({gi['score']})")
    evidence_text = "\n".join(evidence_lines)

    context = {
        "name": player_name,
        "team": str(row.get("Team") or ""),
        "comp": str(row.get("Competition") or ""),
        "mins": int(row.get("Minutes")) if pd.notnull(row.get("Minutes")) else None,
        "posg": str(row.get("Position Group") or ""),
        "overall": float(row.get("_score_0_100", np.nan)),
    }

    prompt = f"""
You are writing a short **physical profile** for a football player based ONLY on the labelled evidence below.
You MUST stick to each label's polarity:
- If a group is **High**, describe it positively.
- If a group is **Average**, keep neutral, matter-of-fact language.
- If a group is **Low**, describe it as a limitation. DO NOT spin it positively.

NEVER contradict a label. NEVER claim a strength where the label is Low, or a weakness where the label is High.
Do not invent facts beyond these groups. Do not quote raw numbers or percentiles.

Player: {context['name']}
Role: {context['posg']}
Team/League: {context['team']} | {context['comp']}
Minutes: {context['mins']}
Composite (0–100): {context['overall']:.0f} if not NaN else "n/a"

Evidence (group → label (score)):
{evidence_text}

Write 4–6 sentences. Cover groups with **High** or **Low** first. Mention **Average** groups only briefly.
End with one crisp summary line of fit (e.g., “Profiles as a high-work-rate wide player with limited top-end speed.”).
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,            # low = faithful to labels
            max_tokens=250,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        # Safe fallback: fully deterministic summary if API fails
        parts = []
        order = ["High", "Low", "Average"]
        # Prioritise High/Low groups in output
        for wanted in ["High", "Low", "Average"]:
            for g, gi in group_info.items():
                if gi["label"] == wanted:
                    if wanted == "High":
                        parts.append(f"Strong {g.lower()}, a clear positive in his profile.")
                    elif wanted == "Low":
                        parts.append(f"Limited {g.lower()}, which reduces impact in that area.")
                    else:
                        parts.append(f"{g} looks steady without standing out.")
        tail = "Overall profile: "
        if pd.notna(context["overall"]):
            tail += f"{context['overall']:.0f}/100 composite."
        else:
            tail += "composite unavailable."
        parts.append(tail)
        return " ".join(parts)

# ========= AI SUMMARY SECTION =========
st.markdown("### 🧠 AI Physical Summary")

# Generate button
if st.button("Generate Physical Report for Selected Player"):
    with st.spinner("Analysing player profile..."):
        summary_text = generate_ai_summary(selected_player, df, percentile_df)
        st.success("✅ Report generated successfully")
        st.markdown(f"**{selected_player} – Summary:**")
        st.write(summary_text)

# ========= RANKING TABLE =========
st.markdown("### Players Ranked by Physical Composite (0–100)")
rank_cols = ["Player", "Team", "Competition", "Position Group Normalised", "Minutes", "_score_0_100_global"] + RADAR_METRICS
table = df[rank_cols].copy()
table = table.sort_values("_score_0_100_global", ascending=False).reset_index(drop=True)
table.index = np.arange(1, len(table) + 1)
table.index.name = "Rank"
st.dataframe(table, use_container_width=True)
