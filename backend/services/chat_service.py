
from sqlalchemy.orm import Session

# LangGraph workflow
from backend.agents.graph import agent_graph

# Database repository functions
from backend.repositories.conversation_repository import (
    get_or_create_conversation
)
from backend.repositories.message_repository import (
    save_message,
    get_conversation_messages
)

# Source deduplication helper
from backend.services.source_service import get_unique_reranked_chunks

# Langfuse tracing
from langfuse import get_client


# Initialize Langfuse client
langfuse = get_client()


def generate_chat_response(
    db: Session,
    user_id: str,
    conversation_id: str,
    user_query: str
) -> dict:

    # --- STEP 1: Get or create conversation ---
    conversation = get_or_create_conversation(
        db=db,
        user_id=user_id,
        conversation_id=conversation_id
    )

    # --- STEP 2: Read the last 20 messages ---
    previous_messages = get_conversation_messages(
        db=db,
        conversation_id=conversation.id,
        limit=20
    )

    # Convert database messages into LLM-compatible dictionaries
    chat_history = [
        {
            "role": message.role,
            "content": message.content
        }
        for message in previous_messages
    ]

    # --- STEP 3: Save the user's new question ---
    save_message(
        db=db,
        conversation_id=conversation.id,
        role="user",
        content=user_query
    )

    # --- STEP 4: Set conversation title if it is new ---
    if conversation.title in ("New Conversation", "New Chat"):

        title = user_query.strip()

        conversation.title = (
            title[:45] + "..."
            if len(title) > 45
            else title
        )

        db.commit()
        db.refresh(conversation)

    # --- STEP 5: Prepare initial LangGraph state ---
    initial_state = {
        "user_id": str(user_id),
        "conversation_id": str(conversation.id),
        "user_query": user_query,
        "chat_history": chat_history,
        "retry_count": 0,
        "current_task_index": 0,
        "completed_tasks": [],
        "execution_plan": [],
        "retrieval_complete": False
    }

    # --- STEP 6: Execute LangGraph with Langfuse tracing ---
    with langfuse.start_as_current_observation(
        as_type="chain",
        name="langgraph-execution",
        input={
            "conversation_id": str(conversation.id),
            "user_id": str(user_id),
            "query": user_query
        }
    ) as graph_observation:

        graph_result = agent_graph.invoke(initial_state)

        graph_observation.update(
            output={
                "status": "completed",
                "final_answer_generated": bool(
                    graph_result.get("final_answer")
                    or graph_result.get("generated_answer")
                ),
                "retrieved_document_count": len(
                    graph_result.get("retrieved_documents") or []
                ),
                "reranked_chunk_count": len(
                    graph_result.get("reranked_chunks") or []
                )
            }
        )

    # --- STEP 7: Extract final answer ---
    answer = (
        graph_result.get("final_answer")
        or graph_result.get("generated_answer")
        or "I couldn't generate a response. Please try again."
    )

    # --- STEP 8: Extract reranked chunks ---
    reranked_chunks = graph_result.get(
        "reranked_chunks",
        []
    ) or []

    # --- STEP 9: Remove duplicate chunks ---
    unique_chunks = get_unique_reranked_chunks(reranked_chunks)

    # --- STEP 10: Build citation/source list ---
    sources = []

    for chunk in unique_chunks:

        metadata = chunk.get("meta") or {}

        source_path = metadata.get(
            "source",
            "Unknown document"
        )

        # Extract filename from source path
        document_name = (
            str(source_path)
            .replace("\\", "/")
            .split("/")[-1]
        )

        page_number = metadata.get("page")

        # Convert zero-based page number to one-based page number
        if page_number is not None:

            try:
                page_number = int(page_number) + 1

            except (TypeError, ValueError):
                pass

        sources.append({
            "document_name": document_name,
            "page_number": page_number,
            "content": chunk.get("text", "")[:300],
            "score": float(chunk.get("score", 0.0))
        })

    # --- STEP 11: Save assistant response and sources ---
    save_message(
        db=db,
        conversation_id=conversation.id,
        role="assistant",
        content=answer,
        sources=sources
    )

    # --- STEP 12: Return result to API caller ---
    return {
        "answer": answer,
        "sources": sources,
        "status": "success"
    }
