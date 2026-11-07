from pydantic import BaseModel, Field
from typing import List


class ChatRequest(BaseModel):

    conversation_id: str
    user_id: str
    query: str = Field(min_length=1, max_length=10000)


class SourceReference(BaseModel):

    document_name: str
    page_number: int | None = None
    content: str | None = None
    score: float | None = None


class ChatResponse(BaseModel):

    conversation_id: str
    answer: str
    sources: List[SourceReference] = Field(default_factory=list)
    status: str