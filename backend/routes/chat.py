from fastapi import APIRouter

from backend.schemas.chat import ChatRequest, ChatResponse
from backend.schemas.chat_service import generate_chat_response


router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post("/query", response_model=ChatResponse)
def query_chat(request: ChatRequest) -> ChatResponse:
    answer = generate_chat_response(request.query)
    return ChatResponse(
        conversation_id=request.conversation_id,
        answer=answer,
        status="success",
    )
