# ============================================================
# ⚽ SHADOW TEAM PAGE
# ============================================================

import streamlit as st
import pandas as pd
from auth import check_password
from branding import show_branding
from lib.shadow_team_repo import list_shadow_team, delete_shadow_team, upsert_shadow_team

# ---------- Authentication ----------
if not check_password():
    st.stop()

# ---------- Branding ----------
show_branding()
st.title("⚽ Shadow Team (4-3-3)")

rows = list_shadow_team()
if not rows:
    st.info("No players added to the shadow team yet.")
    st.stop()

df = pd.DataFrame(rows).sort_values(["position_slot", "rank"]).reset_index(drop=True)

# ---------- Editable order ----------
st.markdown("### Current Shadow Team")
edited_df = st.data_editor(
    df,
    use_container_width=True,
    column_config={
        "player": "Player",
        "position_slot": st.column_config.SelectboxColumn(
            "Position Slot",
            options=["GK","RB","RCB","LCB","LB","CDM","RCM","LCM","RW","ST","LW"],
        ),
        "rank": st.column_config.NumberColumn("Rank", help="Lower = higher in order"),
        "notes": "Notes",
    },
    hide_index=True,
    key="shadow_team_editor"
)

# ---------- Save changes ----------
if st.button("💾 Save Changes"):
    for _, row in edited_df.iterrows():
        upsert_shadow_team(row.to_dict())
    st.success("Shadow team updated successfully.")

# ---------- Remove players ----------
st.markdown("### Remove Player")
to_remove = st.selectbox("Select a player to remove", df["player"].tolist())
if st.button("🗑 Remove Player"):
    delete_shadow_team(to_remove)
    st.rerun()
