
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database.models import Conversation


def get_or_create_conversation(
    db: Session,
    user_id: str,
    conversation_id: str
):
    """
    Retrieve an existing conversation or create a new one.
    """

    # Convert string ID into UUID
    conversation_uuid = uuid.UUID(conversation_id)

    # Check whether conversation already exists
    statement = select(Conversation).where(
        Conversation.id == conversation_uuid
    )

    conversation = db.execute(statement).scalar_one_or_none()

    # If it exists, verify ownership
    if conversation:

        if conversation.user_id != user_id:
            raise PermissionError(
                "This conversation does not belong to this user."
            )

        return conversation

    # Otherwise, create a new conversation
    conversation = Conversation(
        id=conversation_uuid,
        user_id=user_id
    )

    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    return conversation

def get_user_conversations(
    db: Session,
    user_id: str
):
    """
    Retrieve all conversations belonging to a user,
    ordered by most recently updated.
    """

    statement = (
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
    )

    conversations = db.execute(statement).scalars().all()

    return conversations


def get_user_conversation(
    db: Session,
    user_id: str,
    conversation_id: uuid.UUID
):
    """
    Retrieve a conversation only if it belongs to the specified user.
    Does not create a new conversation.
    """

    statement = (
        select(Conversation)
        .where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id
        )
    )

    return db.execute(statement).scalar_one_or_none()
