
import streamlit as st

from chat.chat_state import (
    get_active_conversation,
    add_message
)

from chat.response_service import generate_response


def render_chat_interface():

    conversation = get_active_conversation()

    # --------------------------------------------------
    # Step 1: Handle empty conversation
    # --------------------------------------------------

    if conversation is None:

        st.title("📚 AI Documents Assistant")

        st.markdown("### Welcome! 👋")

        st.write(
            "Create a new chat to start exploring "
            "your documents."
        )

        st.info(
            "Click '+ New Chat' in the sidebar "
            "to start a conversation."
        )

        return

    # --------------------------------------------------
    # Step 2: Display conversation title
    # --------------------------------------------------

    st.title(conversation["title"])
    st.divider()

    # --------------------------------------------------
    # Step 3: Display previous messages
    # --------------------------------------------------

    for message in conversation["messages"]:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            # Display sources if available
            if message.get("sources"):

                with st.expander("View source documents"):

                    for index, source in enumerate(
                        message["sources"],
                        start=1
                    ):

                        st.markdown(
                            f"**Source {index}: "
                            f"{source['document_name']}**"
                        )

                        st.write(
                            f"Page: {source.get('page_number', 'N/A')}"
                        )

                        st.caption(
                            source.get("content", "")
                        )

                        st.divider()

    # --------------------------------------------------
    # Step 4: Accept user question
    # --------------------------------------------------

    user_query = st.chat_input(
        "Ask your question...",
        key=f"chat_input_{conversation['conversation_id']}"
    )

    if not user_query:
        return

    # --------------------------------------------------
    # Step 5: Store user message
    # --------------------------------------------------

    add_message(
        role="user",
        content=user_query
    )

    # --------------------------------------------------
    # Step 6: Call FastAPI backend
    # --------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Searching documents and generating answer..."):

            try:

                result = generate_response(
                    user_query=user_query,
                    conversation_id=conversation["conversation_id"],
                    user_id=st.session_state.chat_owner
                )

                answer = result["answer"]
                sources = result.get("sources", [])

                # Display generated answer
                st.markdown(answer)

                # Display source documents
                if sources:

                    with st.expander("View source documents"):

                        for index, source in enumerate(
                            sources,
                            start=1
                        ):

                            st.markdown(
                                f"**Source {index}: "
                                f"{source['document_name']}**"
                            )

                            st.write(
                                f"Page: {source.get('page_number', 'N/A')}"
                            )

                            st.caption(
                                source.get("content", "")
                            )

                            st.divider()

                # Store assistant response and sources
                add_message(
                    role="assistant",
                    content=answer,
                    sources=sources
                )
                # Refresh the sidebar conversation list on the next rerun
                st.session_state.conversations_loaded = False

            except RuntimeError as e:

                st.error(str(e))

            except (KeyError, ValueError) as e:

                st.error(
                    f"Unexpected backend response: {e}"
                )

            except Exception as e:

                st.error(
                    f"An unexpected error occurred: {e}"
                )

    # Refresh the interface
    st.rerun()
