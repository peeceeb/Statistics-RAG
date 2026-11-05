
from retrieval.hybrid_search import (
    search_documents,
    rerank_documents
)

from backend.services.llm_service import generate_answer


def generate_chat_response(user_query: str) -> dict:
    """
    Orchestrates the complete RAG pipeline:

    1. Retrieve relevant documents from Qdrant.
    2. Rerank retrieved documents using FlashRank.
    3. Generate an answer using OpenAI.
    4. Prepare unique source references.
    """

    # --------------------------------------------------
    # Step 1: Retrieve relevant documents
    # --------------------------------------------------

    retrieved_documents = search_documents(user_query)

    # --------------------------------------------------
    # Step 2: Rerank retrieved documents
    # --------------------------------------------------

    reranked_chunks = rerank_documents(
        user_query,
        retrieved_documents
    )

    # --------------------------------------------------
    # Step 3: Generate answer using OpenAI
    # --------------------------------------------------

    answer = generate_answer(
        user_query=user_query,
        retrieved_chunks=reranked_chunks
    )

    # --------------------------------------------------
    # Step 4: Prepare unique source references
    # --------------------------------------------------

    sources = []
    seen_chunk_ids = set()

    for chunk in reranked_chunks:

        metadata = chunk.get("meta", {})

        chunk_id = metadata.get("chunk_id")

        # Skip duplicate chunks
        if chunk_id and chunk_id in seen_chunk_ids:
            continue

        if chunk_id:
            seen_chunk_ids.add(chunk_id)

        source_path = metadata.get(
            "source",
            "Unknown document"
        )

        document_name = (
            source_path.replace("\\", "/").rsplit("/", 1)[-1]
        )

        page_number = metadata.get("page")

        # Convert zero-based PDF page index
        # into human-readable page number
        if page_number is not None:
            page_number = int(page_number) + 1

        sources.append({
            "document_name": document_name,
            "page_number": page_number,
            "content": chunk.get("text", "")[:300],
            "score": float(chunk.get("score", 0.0))
        })

    # --------------------------------------------------
    # Step 5: Return final response
    # --------------------------------------------------

    return {
        "answer": answer,
        "sources": sources,
        "status": "success"
    }
