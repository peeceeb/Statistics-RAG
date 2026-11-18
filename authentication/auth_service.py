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

def get_id_token() -> str:
    """
    Retrieve the Google ID token for the authenticated user.
    """

    if not is_authenticated():
        raise RuntimeError("User is not authenticated.")

    token = st.user.tokens.get("id")

    if not token:
        raise RuntimeError(
            "Google ID token is unavailable. "
            "Check Streamlit expose_tokens configuration."
        )

    return token