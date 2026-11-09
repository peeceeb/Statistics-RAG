
from sqlalchemy.orm import Session

from backend.agents.graph import agent_graph

from backend.repositories.conversation_repository import (
    get_or_create_conversation
)

from backend.repositories.message_repository import (
    save_message,
    get_conversation_messages
)

from backend.services.source_service import get_unique_reranked_chunks


def generate_chat_response(
    db: Session,
    user_id: str,
    conversation_id: str,
    user_query: str
) -> dict:
    """
    Orchestrates the chat pipeline using LangGraph.

    1. Create or retrieve conversation.
    2. Retrieve previous conversation history.
    3. Save new user message.
    4. Update conversation title.
    5. Execute the LangGraph agent workflow.
    6. Extract the final answer.
    7. Prepare unique source references.
    8. Save assistant response and sources.
    9. Return final response.
    """

    # Step 1: Create or retrieve conversation.
    conversation = get_or_create_conversation(
        db=db,
        user_id=user_id,
        conversation_id=conversation_id
    )

    # Step 2: Retrieve previous conversation history.
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

    # Step 3: Save new user message.
    save_message(
        db=db,
        conversation_id=conversation.id,
        role="user",
        content=user_query
    )

    # Step 4: Update conversation title.
    if conversation.title in ("New Conversation", "New Chat"):

        title = user_query.strip()

        conversation.title = (
            title[:45] + "..."
            if len(title) > 45
            else title
        )

        db.commit()
        db.refresh(conversation)

    # Step 5: Prepare initial LangGraph state.
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

    # Step 6: Execute LangGraph workflow.
    graph_result = agent_graph.invoke(initial_state)

    # Step 7: Extract final answer.
    answer = (
        graph_result.get("final_answer")
        or graph_result.get("generated_answer")
        or "I couldn't generate a response. Please try again."
    )

    # Step 8: Extract reranked chunks.
    reranked_chunks = graph_result.get(
        "reranked_chunks",
        []
    ) or []

    # Step 9: Use the same deduplication logic as response generation.
    unique_chunks = get_unique_reranked_chunks(reranked_chunks)

    # Step 10: Prepare source references.
    sources = []

    for chunk in unique_chunks:

        metadata = chunk.get("meta") or {}

        source_path = metadata.get(
            "source",
            "Unknown document"
        )

        document_name = (
            str(source_path)
            .replace("\\", "/")
            .split("/")[-1]
        )

        page_number = metadata.get("page")

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

    # Step 11: Save assistant response and sources.
    save_message(
        db=db,
        conversation_id=conversation.id,
        role="assistant",
        content=answer,
        sources=sources
    )

    # Step 12: Return final response.
    return {
        "answer": answer,
        "sources": sources,
        "status": "success"
    }
