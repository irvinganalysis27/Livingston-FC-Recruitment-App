import streamlit as st
from lib.session_manager import save_ui_state, get_ui_state

def ui_multiselect(label, options, key, default=None, **kwargs):
    value = get_ui_state(key, default)
    new_val = st.multiselect(label, options, default=value, key=key, **kwargs)
    save_ui_state(**{key: new_val})
    return new_val

def ui_toggle(label, key, default=False, **kwargs):
    value = get_ui_state(key, default)
    new_val = st.toggle(label, value=value, key=key, **kwargs)
    save_ui_state(**{key: new_val})
    return new_val

def ui_selectbox(label, options, key, default=None, **kwargs):
    value = get_ui_state(key, default)
    if value not in options and options:
        value = options[0]
    new_val = st.selectbox(label, options, index=options.index(value) if value in options else 0, key=key, **kwargs)
    save_ui_state(**{key: new_val})
    return new_val
