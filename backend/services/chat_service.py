
from sqlalchemy.orm import Session

from retrieval.hybrid_search import (
    search_documents,
    rerank_documents
)

from backend.services.llm_service import generate_answer

from backend.repositories.conversation_repository import (
    get_or_create_conversation
)

from backend.repositories.message_repository import (
    save_message,
    get_conversation_messages
)


def generate_chat_response(
    db: Session,
    user_id: str,
    conversation_id: str,
    user_query: str
) -> dict:
    """
    Orchestrates the complete chat and RAG pipeline.

    1. Create or retrieve conversation.
    2. Retrieve previous conversation history.
    3. Save new user message.
    4. Update conversation title.
    5. Retrieve relevant documents.
    6. Rerank retrieved documents.
    7. Generate answer using history and documents.
    8. Prepare unique source references.
    9. Save assistant response and sources.
    10. Return final response.
    """

    # --------------------------------------------------
    # Step 1: Create or retrieve conversation
    # --------------------------------------------------

    conversation = get_or_create_conversation(
        db=db,
        user_id=user_id,
        conversation_id=conversation_id
    )

    # --------------------------------------------------
    # Step 2: Retrieve previous conversation history
    # --------------------------------------------------

    previous_messages = get_conversation_messages(
        db=db,
        conversation_id=conversation.id,
        limit=20
    )

    chat_history = [
        {
            "role": message.role,
            "content": message.content
        }
        for message in previous_messages
    ]

    # --------------------------------------------------
    # Step 3: Save new user message
    # --------------------------------------------------

    save_message(
        db=db,
        conversation_id=conversation.id,
        role="user",
        content=user_query
    )

    # --------------------------------------------------
    # Step 4: Update conversation title
    # --------------------------------------------------

    if conversation.title in ("New Conversation", "New Chat"):

        title = user_query.strip()

        conversation.title = (
            title[:45] + "..."
            if len(title) > 45
            else title
        )

        db.commit()
        db.refresh(conversation)

    # --------------------------------------------------
    # Step 5: Retrieve relevant documents
    # --------------------------------------------------

    retrieved_documents = search_documents(user_query)

    # --------------------------------------------------
    # Step 6: Rerank retrieved documents
    # --------------------------------------------------

    reranked_chunks = rerank_documents(
        user_query,
        retrieved_documents
    )

    # --------------------------------------------------
    # Step 7: Generate answer using history + RAG
    # --------------------------------------------------

    answer = generate_answer(
        user_query=user_query,
        retrieved_chunks=reranked_chunks,
        chat_history=chat_history
    )

    # --------------------------------------------------
    # Step 8: Prepare unique source references
    # --------------------------------------------------

    sources = []
    seen_chunk_ids = set()

    for chunk in reranked_chunks:

        metadata = chunk.get("meta", {})

        chunk_id = metadata.get("chunk_id")

        if chunk_id and chunk_id in seen_chunk_ids:
            continue

        if chunk_id:
            seen_chunk_ids.add(chunk_id)

        source_path = metadata.get(
            "source",
            "Unknown document"
        )

        document_name = (
            source_path.replace("\\", "/").split("/")[-1]
        )

        page_number = metadata.get("page")

        if page_number is not None:
            page_number = int(page_number) + 1

        sources.append({
            "document_name": document_name,
            "page_number": page_number,
            "content": chunk.get("text", "")[:300],
            "score": float(chunk.get("score", 0.0))
        })

    # --------------------------------------------------
    # Step 9: Save assistant response and sources
    # --------------------------------------------------

    save_message(
        db=db,
        conversation_id=conversation.id,
        role="assistant",
        content=answer,
        sources=sources
    )

    # --------------------------------------------------
    # Step 10: Return final response
    # --------------------------------------------------

    return {
        "answer": answer,
        "sources": sources,
        "status": "success"
    }
