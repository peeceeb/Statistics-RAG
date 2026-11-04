import streamlit as st
import requests

from chat.chat_state import (
    get_active_conversation,
    add_message
)

from config.settings import BACKEND_API_URL


def call_chat_api(user_query, conversation):

    """
    Send the user's question to FastAPI.
    """

    url = f"{BACKEND_API_URL}/api/v1/chat/query"

    payload = {

        "conversation_id": conversation["conversation_id"],

        "user_id": st.session_state.chat_owner,

        "query": user_query

    }

    response = requests.post(
        url,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    return response.json()["answer"]


def render_chat_interface():

    conversation = get_active_conversation()

    if conversation is None:

        st.title("📚 Regulatory AI Assistant")

        st.markdown("### Welcome! 👋")

        st.write(
            "Create a new chat to start exploring "
            "your regulatory documents."
        )

        st.info(
            "Click '+ New Chat' in the sidebar "
            "to start a conversation."
        )

        return

    st.title(conversation["title"])

    st.divider()

    # Display existing messages
    for message in conversation["messages"]:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # Chat input
    user_query = st.chat_input(
        "Ask your regulatory question...",
        key=f"chat_input_{conversation['conversation_id']}"
    )

    if user_query:

        # Store user message
        add_message(
            role="user",
            content=user_query
        )

        # Call FastAPI
        with st.chat_message("assistant"):

            with st.spinner("AI is thinking..."):

                try:

                    answer = call_chat_api(
                        user_query,
                        conversation
                    )

                    add_message(
                        role="assistant",
                        content=answer
                    )

                    st.markdown(answer)

                except requests.exceptions.ConnectionError:

                    error_message = (
                        "Unable to connect to the backend. "
                        "Please check whether FastAPI is running."
                    )

                    st.error(error_message)

                except requests.exceptions.RequestException as e:

                    st.error(f"API request failed: {e}")

        st.rerun()