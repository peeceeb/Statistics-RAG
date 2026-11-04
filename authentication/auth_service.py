import streamlit as st


def is_authenticated() -> bool:
    """
    Check whether the user is logged in.
    """

    return st.user.is_logged_in


def login():
    """
    Initiate Google authentication.
    """

    st.login()


def logout():
    """
    Log the user out.
    """

    st.logout()


def get_user_details() -> dict:
    """
    Retrieve authenticated user information.
    """

    if not is_authenticated():
        return {}

    return {
        "user_id": st.user.get("sub"),
        "name": st.user.get("name"),
        "email": st.user.get("email"),
        "picture": st.user.get("picture")
    }