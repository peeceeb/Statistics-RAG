import streamlit as st

from uuid import uuid4
from datetime import datetime


def initialize_chat_state(user_id: str):

    if st.session_state.get("chat_owner") != user_id:

        st.session_state.chat_owner = user_id

        st.session_state.conversations = {}

        st.session_state.active_conversation_id = None


def create_new_chat():

    conversation_id = str(uuid4())

    st.session_state.conversations[conversation_id] = {

        "conversation_id": conversation_id,

        "title": "New Chat",

        "created_at": datetime.now().isoformat(),

        "updated_at": datetime.now().isoformat(),

        "messages": []

    }

    st.session_state.active_conversation_id = conversation_id


def select_chat(conversation_id: str):

    if conversation_id in st.session_state.conversations:

        st.session_state.active_conversation_id = conversation_id


def get_active_conversation():

    conversation_id = st.session_state.active_conversation_id

    if conversation_id:

        return st.session_state.conversations.get(conversation_id)

    return None


def add_message(role: str, content: str):

    conversation = get_active_conversation()

    if conversation is None:
        return

    # Add the message
    conversation["messages"].append({

        "role": role,

        "content": content,

        "timestamp": datetime.now().isoformat()

    })

    # Update conversation title using first user message
    if role == "user" and conversation["title"] == "New Chat":

        conversation["title"] = (
            content[:45] + "..."
            if len(content) > 45
            else content
        )

    # Update last activity
    conversation["updated_at"] = datetime.now().isoformat()