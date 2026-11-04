import streamlit as st

from authentication.auth_service import (
    is_authenticated,
    get_user_details
)

from authentication.login_page import show_login_page

from chat.chat_state import initialize_chat_state

from chat.sidebar import render_sidebar

from chat.chat_interface import render_chat_interface

from config.settings import (
    APP_NAME,
    APP_ICON,
    PAGE_LAYOUT
)


# Configure application
st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT
)


# Authentication
if not is_authenticated():

    show_login_page()

    st.stop()


# Get user information
user = get_user_details()


# Initialize chat state
initialize_chat_state(user["user_id"])


# Render sidebar
render_sidebar(user)


# Render chat interface
render_chat_interface()