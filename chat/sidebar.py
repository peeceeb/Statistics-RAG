import streamlit as st

from chat.chat_state import (
    create_new_chat,
    select_chat
)

from authentication.auth_service import logout


def render_sidebar(user):

    with st.sidebar:

        st.title("📚 Regulatory AI")

        st.divider()

        # New Chat
        st.button(
            "➕ New Chat",
            use_container_width=True,
            on_click=create_new_chat
        )

        st.divider()

        st.subheader("Recent Chats")

        conversations = st.session_state.conversations

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

                    select_chat(conversation_id)

        else:

            st.caption("No conversations yet.")

        st.divider()

        # User Profile
        st.subheader("User Profile")

        if user.get("picture"):

            st.image(
                user["picture"],
                width=80
            )

        st.write(user.get("name", "User"))

        st.caption(user.get("email", ""))

        st.button(
            "Logout",
            use_container_width=True,
            on_click=logout
        )