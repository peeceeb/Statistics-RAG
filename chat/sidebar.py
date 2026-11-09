
import streamlit as st

from chat.chat_state import (
    create_new_chat,
    select_chat,
    load_conversations,
    load_conversation_messages,
    clear_recent_chats
)

from chat.response_service import (
    get_conversations,
    get_conversation_messages
)

from authentication.auth_service import logout


def render_sidebar(user):

    user_id = st.session_state.chat_owner

    with st.sidebar:

        st.title("📚 Regulatory AI")
        st.divider()

        # Create a new conversation
        st.button(
            "➕ New Chat",
            use_container_width=True,
            on_click=create_new_chat
        )

        st.divider()
        st.subheader("Recent Chats")

        # Clear recent chats
        st.button(
            "🗑️ Clear Recent Chats",
            use_container_width=True,
            on_click=clear_recent_chats
        )

        # Fetch conversations when required
        if (
            st.session_state.get("conversations_loaded_for_user") != user_id
            or not st.session_state.get("conversations_loaded", False)
        ):
            try:
                conversations = get_conversations(user_id)
                load_conversations(conversations)

                st.session_state.conversations_loaded_for_user = user_id
                st.session_state.conversations_loaded = True

            except RuntimeError as e:
                st.error(str(e))

        # Filter hidden conversations
        hidden_ids = st.session_state.get(
            "hidden_recent_chat_ids", set()
        )

        conversations = {
            conversation_id: conversation
            for conversation_id, conversation
            in st.session_state.conversations.items()
            if conversation_id not in hidden_ids
        }

        # Display recent conversations
        if conversations:

            sorted_conversations = sorted(
                conversations.values(),
                key=lambda x: x.get(
                    "updated_at",
                    x["created_at"]
                ),
                reverse=True
            )

            for conversation in sorted_conversations:

                conversation_id = conversation["conversation_id"]
                title = conversation["title"]

                is_active = (
                    conversation_id
                    == st.session_state.active_conversation_id
                )

                if st.button(
                    f"{'📌 ' if is_active else '💬 '}{title}",
                    key=f"chat_{conversation_id}",
                    use_container_width=True
                ):

                    try:
                        select_chat(conversation_id)

                        messages = get_conversation_messages(
                            user_id=user_id,
                            conversation_id=conversation_id
                        )

                        load_conversation_messages(
                            conversation_id=conversation_id,
                            messages=messages
                        )

                    except RuntimeError as e:
                        st.error(str(e))

        else:
            st.caption("No conversations yet.")

        st.divider()

        # User profile
        st.subheader("User Profile")

        if user.get("picture"):
            st.image(user["picture"], width=80)

        st.write(user.get("name", "User"))
        st.caption(user.get("email", ""))

        st.button(
            "Logout",
            use_container_width=True,
            on_click=logout
        )
