
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    MessageResponse
)

from backend.database.connection import get_db

from backend.services.chat_service import generate_chat_response

from backend.repositories.conversation_repository import (
    get_user_conversations,
    get_or_create_conversation
)

from backend.repositories.message_repository import (
    get_conversation_messages
)


router = APIRouter(
    prefix="/api/v1/chat",
    tags=["chat"]
)


# --------------------------------------------------
# 1. Generate Chat Response
# --------------------------------------------------

@router.post("/query", response_model=ChatResponse)
def query_chat(
    request: ChatRequest,
    db: Session = Depends(get_db)
) -> ChatResponse:

    try:

        result = generate_chat_response(
            db=db,
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            user_query=request.query
        )

        return ChatResponse(
            conversation_id=request.conversation_id,
            answer=result["answer"],
            sources=result["sources"],
            status=result["status"]
        )

    except PermissionError as exc:

        raise HTTPException(
            status_code=403,
            detail=str(exc)
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )


# --------------------------------------------------
# 2. Retrieve User Conversations
# --------------------------------------------------

@router.get(
    "/conversations",
    response_model=list[ConversationResponse]
)
def list_conversations(
    user_id: str,
    db: Session = Depends(get_db)
):

    conversations = get_user_conversations(
        db=db,
        user_id=user_id
    )

    return [
        ConversationResponse(
            conversation_id=str(conversation.id),
            title=conversation.title,
            created_at=conversation.created_at.isoformat(),
            updated_at=conversation.updated_at.isoformat()
        )
        for conversation in conversations
    ]


# --------------------------------------------------
# 3. Retrieve Messages from a Conversation
# --------------------------------------------------

@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[MessageResponse]
)
def list_conversation_messages(
    conversation_id: str,
    user_id: str,
    db: Session = Depends(get_db)
):

    try:

        # Validate conversation ID
        conversation_uuid = UUID(conversation_id)

        # Verify conversation ownership
        conversation = get_or_create_conversation(
            db=db,
            user_id=user_id,
            conversation_id=str(conversation_uuid)
        )

        if conversation is None:
            raise HTTPException(
            status_code=404,
            detail="Conversation not found."
            )

        messages = get_conversation_messages(
            db=db,
            conversation_id=conversation.id,
            limit=20
        )

        return [
            MessageResponse(
                role=message.role,
                content=message.content,
                sources=message.sources or [],
                created_at=message.created_at.isoformat()
            )
            for message in messages
        ]

    except PermissionError as exc:

        raise HTTPException(
            status_code=403,
            detail=str(exc)
        )

    except ValueError:

        raise HTTPException(
            status_code=400,
            detail="Invalid conversation ID."
        )
