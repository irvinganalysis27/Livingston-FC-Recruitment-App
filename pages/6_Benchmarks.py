import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime

from auth import check_password
from branding import show_branding

# ============================================================
# Setup & Protection
# ============================================================
if not check_password():
    st.stop()

show_branding()
st.title("📊 Position Benchmarks")

# ============================================================
# Paths & Config
# ============================================================
APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent
DATA_PATH = ROOT_DIR / "statsbomb_player_stats_clean.csv"

# ============================================================
# Position Mapping (same as other pages)
# ============================================================
RAW_TO_GROUP = {
    "LEFTBACK": "Full Back", "LEFTWINGBACK": "Full Back",
    "RIGHTBACK": "Full Back", "RIGHTWINGBACK": "Full Back",
    "CENTREBACK": "Centre Back", "LEFTCENTREBACK": "Centre Back", "RIGHTCENTRECENTREBACK": "Centre Back",
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
    if pd.isna(tok):
        return ""
    t = str(tok).upper().strip()
    t = re.sub(r"[.\-_/]", " ", t)
    return re.sub(r"\s+", "", t)

def map_first_position_to_group(primary_pos_cell) -> str:
    return RAW_TO_GROUP.get(_clean_pos_token(primary_pos_cell), None)

# ============================================================
# Load & Preprocess Data
# ============================================================
@st.cache_data(ttl=86400)
def load_statsbomb_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Normalise position column
    if "Primary Position" in df.columns:
        df["Six-Group Position"] = df["Primary Position"].apply(map_first_position_to_group)
    elif "Position" in df.columns:
        df["Six-Group Position"] = df["Position"].apply(map_first_position_to_group)
    else:
        df["Six-Group Position"] = np.nan

    # Standardise minutes column
    if "Minutes" in df.columns:
        df["Minutes played"] = pd.to_numeric(df["Minutes"], errors="coerce").fillna(0)

    return df

# Load dataset
try:
    df_all = load_statsbomb_data(DATA_PATH)
except Exception as e:
    st.error(f"❌ Could not load StatsBomb data: {e}")
    st.stop()

# Filter to eligible players
df_all = df_all[df_all["Minutes played"] >= 600].copy()
if df_all.empty:
    st.warning("No players with 600+ minutes found.")
    st.stop()

# ============================================================
# Metric Filtering
# ============================================================
numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
drop_cols = ["Minutes played", "Age", "Height", "Weight", "Multiplier"]
HIDE_PATTERNS = ["player season", "account id"]

metrics = []
for c in numeric_cols:
    cname = str(c).lower()
    if any(p in cname for p in HIDE_PATTERNS):
        continue
    if c not in drop_cols:
        metrics.append(c)

if not metrics:
    st.error("No valid numeric metrics available for benchmarks.")
    st.stop()

# ============================================================
# Compute Benchmarks (per position)
# ============================================================
@st.cache_data
def compute_benchmarks(df: pd.DataFrame, pos_col: str, metrics: list[str]) -> dict:
    """Return dict of position → metric → percentile bands."""
    benchmarks = {}
    for pos, sub in df.groupby(pos_col):
        if pos is None or str(pos).strip() == "":
            continue
        benchmarks[pos] = {}
        for m in metrics:
            vals = pd.to_numeric(sub[m], errors="coerce").dropna()
            if len(vals) < 10:
                continue
            q10, q30, q70, q90 = np.percentile(vals, [10, 30, 70, 90])
            benchmarks[pos][m] = {
                "Poor (<10%)": f"<{q10:.2f}",
                "Below Average": f"{q10:.2f}–{q30:.2f}",
                "Average": f"{q30:.2f}–{q70:.2f}",
                "Good": f"{q70:.2f}–{q90:.2f}",
                "Excellent (>90%)": f">{q90:.2f}",
                "Sustainable Good Range (30–90%)": f"{q30:.2f}–{q90:.2f}",
            }
    return benchmarks

benchmarks = compute_benchmarks(df_all, "Six-Group Position", metrics)
positions = sorted(list(benchmarks.keys()))

if not positions:
    st.error("No position groups found with enough players to calculate benchmarks.")
    st.stop()

# ============================================================
# UI: Select Position + Metric
# ============================================================
st.markdown("### Select a Position and Metric")

selected_position = st.selectbox("Position", positions)
metric_list = sorted(list(benchmarks[selected_position].keys()))

if not metric_list:
    st.warning(f"No valid metrics for {selected_position}.")
    st.stop()

if "selected_metric" not in st.session_state or st.session_state.selected_metric not in metric_list:
    st.session_state.selected_metric = metric_list[0]

selected_metric = st.selectbox(
    "Metric",
    metric_list,
    index=metric_list.index(st.session_state.selected_metric),
)
st.session_state.selected_metric = selected_metric

row = benchmarks[selected_position][selected_metric]

# ============================================================
# Display Ranges
# ============================================================
st.subheader(f"Benchmark Ranges — {selected_position}: {selected_metric}")
st.markdown(f"<span style='color:red'>Poor:</span> {row['Poor (<10%)']}", unsafe_allow_html=True)
st.markdown(f"<span style='color:orange'>Below Average:</span> {row['Below Average']}", unsafe_allow_html=True)
st.markdown(f"<span style='color:grey'>Average:</span> {row['Average']}", unsafe_allow_html=True)
st.markdown(f"<span style='color:blue'>Good:</span> {row['Good']}", unsafe_allow_html=True)
st.markdown(f"<span style='color:green'>Excellent:</span> {row['Excellent (>90%)']}", unsafe_allow_html=True)

st.write("**Sustainable Good Range (30–90%)**:", row["Sustainable Good Range (30–90%)"])

# ============================================================
# Test a Value
# ============================================================
st.markdown("### Test a Value")
value = st.number_input("Enter a value to test", step=0.01)

def get_category(val, r):
    def parse_range(text):
        if "–" in text:
            parts = text.split("–")
            return float(parts[0]), float(parts[1])
        return None, None

    try:
        poor_upper = float(r["Poor (<10%)"].replace("<", ""))
    except:
        poor_upper = None
    below_low, below_high = parse_range(r["Below Average"])
    avg_low, avg_high = parse_range(r["Average"])
    good_low, good_high = parse_range(r["Good"])
    try:
        exc_threshold = float(r["Excellent (>90%)"].replace(">", ""))
    except:
        exc_threshold = None

    if poor_upper and val < poor_upper:
        return "Poor", "red"
    elif below_low and below_low <= val <= below_high:
        return "Below Average", "orange"
    elif avg_low and avg_low <= val <= avg_high:
        return "Average", "grey"
    elif good_low and good_low <= val <= good_high:
        return "Good", "blue"
    elif exc_threshold and val > exc_threshold:
        return "Excellent", "green"
    return "Out of Range", "black"

if value:
    category, color = get_category(value, row)
    st.markdown(f"**Category:** <span style='color:{color}'>{category}</span>", unsafe_allow_html=True)
