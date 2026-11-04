import streamlit as st

from authentication.auth_service import login


def show_login_page():


    st.markdown(
        "Your intelligent assistant for "
        "Statistical Analysis"
    )

    st.divider()

    st.subheader("Welcome!")

    st.write(
        "Sign in with your Google account "
        "to access the Statistical AI Assistant."
    )

    st.button(
        "Continue with Google",
        on_click=login,
        use_container_width=True
    )

    st.caption(
        "Secure authentication powered by Google."
    )