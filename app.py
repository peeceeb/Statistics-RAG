import streamlit as st
import requests
from io import BytesIO

from authentication.auth_service import (
    is_authenticated,
    get_user_details,
    logout
)

from authentication.login_page import show_login_page

from config.settings import (
    APP_NAME,
    APP_ICON,
    PAGE_LAYOUT
)


# Configure Streamlit page
st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT
)


# Step 1: Check authentication

if not is_authenticated():

    show_login_page()

    st.stop()


# Step 2: Retrieve user information

user = get_user_details()



# Step 3: Display authenticated application

st.title(f"Welcome, {user['name']}!")

st.write(
    "You have successfully logged into "
    "the Statistical AI Assistant."
)


# Step 4: Display user information in sidebar

with st.sidebar:

    st.subheader("User Profile")

    if user["picture"]:
        try:
            response = requests.get(
                user["picture"],
                timeout=10
            )

            response.raise_for_status()

            st.image(
                BytesIO(response.content),
                width=100
            )

        except Exception as e:
            st.warning("Unable to load profile picture.")
            print(f"Image loading error: {e}")

    else:
        st.info("Profile picture not available.")

    st.write(f"Name: {user['name']}")

    st.write(f"Email: {user['email']}")

    st.divider()

    st.button(
        "Logout",
        on_click=logout,
        use_container_width=True
    )


# Step 5: Phase 1 completion message

st.success(
    "Google authentication completed successfully!"
)