
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database.models import Message


def save_message(
    db: Session,
    conversation_id,
    role: str,
    content: str,
    sources: list | None = None
):
    """
    Save a user or assistant message.
    """

    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        sources=sources or []
    )

    db.add(message)
    db.commit()
    db.refresh(message)

    return message


def get_conversation_messages(
    db: Session,
    conversation_id,
    limit: int = 20
):
    """
    Retrieve the most recent messages from a conversation
    in chronological order.
    """

    statement = (
        select(Message)
        .where(
            Message.conversation_id == conversation_id
        )
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(limit)
    )

    messages = (
        db.execute(statement)
        .scalars()
        .all()
    )

    return list(reversed(messages))
