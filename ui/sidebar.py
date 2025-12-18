import streamlit as st


def render_sidebar():
    # Hide default Streamlit multipage navigation
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] { display: none; }
            /* Allow sidebar titles and links to wrap instead of truncating */
            section[data-testid="stSidebar"] * {
                white-space: normal !important;
                word-wrap: break-word !important;
            }

            /* Specifically target page links */
            a[data-testid="stPageLink"] {
                white-space: normal !important;
                line-height: 1.3;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.sidebar:
        st.markdown("## Livingston FC App")

        # ---------- Radars ----------
        with st.expander("📊 Radars"):
            st.page_link("pages/001_Statsbomb_Radar.py", label="StatsBomb Radar")
            st.divider()
            st.page_link("pages/002_Skillcorner_Radar.py", label="SkillCorner Radar")
            st.divider()
            st.page_link("pages/003_Wyscout_Radar.py", label="Wyscout Radar")
            st.divider()
            st.page_link(
                "pages/012_Historical_Statsbomb_Radar.py",
                label="Historical StatsBomb Radar"
            )

        # ---------- Scatter Plots ----------
        with st.expander("📈 Scatter Plots"):
            st.page_link(
                "pages/008_Statsbomb_Scatter_Plot.py",
                label="StatsBomb Scatter Plot"
            )
            st.divider()
            st.page_link(
                "pages/009_SkillCorner_Scatter_Plot.py",
                label="SkillCorner Scatter Plot"
            )

        # ---------- Rankings & Comparison ----------
        with st.expander("🏆 Rankings & Comparison"):
            st.page_link(
                "pages/004_Player_Comparison.py",
                label="Player Comparison"
            )
            st.divider()
            st.page_link(
                "pages/005_Team_Rankings.py",
                label="Team Rankings"
            )

        # ---------- Squad Tools ----------
        with st.expander("🧠 Squad Tools"):
            st.page_link(
                "pages/006_Favourites.py",
                label="Favourites"
            )
            st.divider()
            st.page_link(
                "pages/007_Shadow_Team.py",
                label="Shadow Team"
            )

        # ---------- Other Info ----------
        with st.expander("⚙️ Other Info"):
            st.page_link(
                "pages/010_Weightings.py",
                label="Weightings"
            )
            st.divider()
            st.page_link(
                "pages/011_Benchmarks.py",
                label="Benchmarks"
            )

        st.divider()
