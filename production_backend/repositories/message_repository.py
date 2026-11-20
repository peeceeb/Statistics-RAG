
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database.models import Message


def save_message(
    db: Session,
    conversation_id: str,
    role: str,
    content: str,
    sources: list | None = None
) -> Message:
    """
    Save a user or assistant message to the database.

    Args:
        db: SQLAlchemy database session.
        conversation_id: ID of the conversation.
        role: Message role (user or assistant).
        content: Message content.
        sources: Optional source references.

    Returns:
        The saved Message object.
    """

    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        sources=sources
    )

    db.add(message)
    db.commit()
    db.refresh(message)

    return message


def get_conversation_messages(
    db: Session,
    conversation_id: str,
    limit: int = 20
) -> list[Message]:
    """
    Retrieve the most recent messages for a conversation.

    Messages are returned in chronological order.

    Args:
        db: SQLAlchemy database session.
        conversation_id: ID of the conversation.
        limit: Maximum number of messages to retrieve.

    Returns:
        List of Message objects.
    """

    statement = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(
            Message.created_at.desc(),
            Message.id.desc()
        )
        .limit(limit)
    )

    messages = db.execute(statement).scalars().all()

    return list(reversed(messages))
