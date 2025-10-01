import streamlit as st
from pathlib import Path
from PIL import Image

APP_DIR = Path(__file__).parent
ASSETS_DIR = APP_DIR / "assets"

def open_image(path: Path):
    try:
        return Image.open(path)
    except Exception:
        return None

def show_branding():
    """Displays the Livingston FC Recruitment App header with two logos."""
    logo_path = ASSETS_DIR / "Livingston_FC_club_badge_new.png"
    logo = open_image(logo_path)

    left, mid, right = st.columns([1, 6, 1])

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
