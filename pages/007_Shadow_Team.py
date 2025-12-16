# ============================================================
# ⚽ SHADOW TEAM PAGE (grouped by position)
# ============================================================

import streamlit as st
import pandas as pd
from auth import check_password
from branding import show_branding
from lib.shadow_team_repo import list_shadow_team, delete_shadow_team, upsert_shadow_team

# ---------- Authentication ----------
if not check_password():
    st.stop()

st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Branding ----------
show_branding()
st.title("Shadow Team")

# ---------- Fetch data ----------
rows = list_shadow_team()
if not rows:
    st.info("No players added to the shadow team yet.")
    st.stop()

df = pd.DataFrame(rows).sort_values(["position_slot", "rank"]).reset_index(drop=True)

# ✅ Remove internal and technical columns (id, timestamps)
df = df.drop(columns=[c for c in df.columns if c in ["id", "created_at", "updated_at"]], errors="ignore")

# ---------- Define display order ----------
POSITION_ORDER = ["GK", "RB", "LB", "CB", "6", "8", "10", "RW", "LW", "ST"]
POSITION_LABELS = {
    "GK": "Goalkeeper",
    "RB": "Right Back",
    "LB": "Left Back",
    "CB": "Centre Back",
    "6": "Number 6",
    "8": "Number 8",
    "10": "Number 10",
    "RW": "Right Wing",
    "LW": "Left Wing",
    "ST": "Striker",
}

# ---------- Display grouped tables ----------
st.markdown("### Current Shadow Team")

for pos in POSITION_ORDER:
    pos_df = df[df["position_slot"] == pos]
    if pos_df.empty:
        continue

    st.subheader(POSITION_LABELS.get(pos, pos))
    edited_df = st.data_editor(
        pos_df,
        width="stretch",
        hide_index=True,
        column_config={
            "player": "Player",
            "position_slot": st.column_config.SelectboxColumn(
                "Position Slot",
                options=POSITION_ORDER,
                required=True,
            ),
            "rank": st.column_config.NumberColumn("Rank", help="Lower = higher in order"),
            "notes": st.column_config.TextColumn("Notes"),
        },
        key=f"editor_{pos}",
    )

    # Save updates for this position
    if st.button(f"💾 Save {pos} Changes", key=f"save_{pos}"):
        for _, row in edited_df.iterrows():
            upsert_shadow_team(row.to_dict())
        st.success(f"{pos} group updated successfully.")

st.divider()

# ---------- Remove players ----------
st.markdown("### 🗑 Remove Player")
to_remove = st.selectbox(
    "Select a player to remove",
    df["player"].tolist(),
    index=None,
    placeholder="Choose a player...",
)
if st.button("Remove Player"):
    if to_remove:
        delete_shadow_team(to_remove)
        st.success(f"Removed {to_remove} from Shadow Team.")
        st.rerun()
    else:
        st.warning("Please select a player first.")
