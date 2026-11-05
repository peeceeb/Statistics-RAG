
from fastapi import APIRouter

from backend.schemas.chat import ChatRequest, ChatResponse
from backend.services.chat_service import generate_chat_response


router = APIRouter(
    prefix="/api/v1/chat",
    tags=["chat"]
)


@router.post("/query", response_model=ChatResponse)
def query_chat(request: ChatRequest) -> ChatResponse:

    # Execute the complete RAG pipeline
    result = generate_chat_response(request.query)

    # Construct the API response
    return ChatResponse(
        conversation_id=request.conversation_id,
        answer=result["answer"],
        sources=result["sources"],
        status=result["status"]
    )
