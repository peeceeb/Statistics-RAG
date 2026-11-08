
import streamlit as st
from uuid import uuid4
from datetime import datetime


def initialize_chat_state(user_id: str):
    if st.session_state.get("chat_owner") != user_id:
        st.session_state.chat_owner = user_id
        st.session_state.conversations = {}
        st.session_state.active_conversation_id = None

        # Reset conversation synchronization for the new user
        st.session_state.conversations_loaded = False
        st.session_state.conversations_loaded_for_user = None
        st.session_state.hidden_recent_chat_ids = set()


def create_new_chat():
    conversation_id = str(uuid4())
    current_time = datetime.now().isoformat()

    st.session_state.conversations[conversation_id] = {
        "conversation_id": conversation_id,
        "title": "New Chat",
        "created_at": current_time,
        "updated_at": current_time,
        "messages": []
    }

    st.session_state.active_conversation_id = conversation_id


def select_chat(conversation_id: str):
    if conversation_id in st.session_state.conversations:
        st.session_state.active_conversation_id = conversation_id

def clear_recent_chats():
    """
    Hide existing conversations from the sidebar.
    Does not delete anything from PostgreSQL.
    """

    hidden_ids = st.session_state.get(
        "hidden_recent_chat_ids", set()
    )

    hidden_ids.update(
        st.session_state.conversations.keys()
    )

    st.session_state.hidden_recent_chat_ids = hidden_ids
    st.session_state.active_conversation_id = None

def load_conversations(conversations: list):
    """
    Load conversations retrieved from FastAPI
    into Streamlit session state.
    """
    for conversation in conversations:
        conversation_id = conversation["conversation_id"]

        existing = st.session_state.conversations.get(
            conversation_id, {}
        )

        st.session_state.conversations[conversation_id] = {
            "conversation_id": conversation_id,
            "title": conversation["title"],
            "created_at": conversation["created_at"],
            "updated_at": conversation["updated_at"],
            "messages": existing.get("messages", [])
        }


def load_conversation_messages(conversation_id: str, messages: list):
    """
    Load messages retrieved from FastAPI
    into a specific conversation.
    """
    conversation = st.session_state.conversations.get(conversation_id)

    if conversation is None:
        return

    conversation["messages"] = [
        {
            "role": message["role"],
            "content": message["content"],
            "sources": message.get("sources", []),
            "timestamp": message["created_at"]
        }
        for message in messages
    ]

    st.session_state.active_conversation_id = conversation_id


def get_active_conversation():
    conversation_id = st.session_state.active_conversation_id

    if conversation_id:
        return st.session_state.conversations.get(conversation_id)

    return None


def add_message(
    role: str,
    content: str,
    sources: list = None
):
    conversation = get_active_conversation()

    if conversation is None:
        return

    current_time = datetime.now().isoformat()

    # Add message and source references
    conversation["messages"].append({
        "role": role,
        "content": content,
        "sources": sources or [],
        "timestamp": current_time
    })

    # Update conversation title using first user message
    if role == "user" and conversation["title"] == "New Chat" and content.strip():
        conversation["title"] = (
            content[:45] + "..."
            if len(content) > 45
            else content
        )

    # Update last activity
    conversation["updated_at"] = current_time
