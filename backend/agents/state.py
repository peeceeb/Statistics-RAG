
from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared state passed between nodes during one LangGraph execution.

    Each node reads the current state and returns the fields it updates.
    """

    # 1. User and conversation information
    user_id: str
    conversation_id: str
    user_query: str
    chat_history: list[dict[str, str]]

    # 2. Query understanding
    intent: str
    query_complexity: str
    requires_planning: bool

    # 3. Planning and task tracking
    execution_plan: list[dict[str, Any]]
    current_task: dict[str, Any]
    current_task_index: int
    completed_tasks: list[dict[str, Any]]

    # 4. Document retrieval and reranking
    retrieved_documents: list[Any]
    reranked_chunks: list[dict[str, Any]]

    # 5. Specialist agent outputs
    analysis_result: str
    comparison_result: str
    summary_result: str

    # 6. Response generation
    generated_answer: str
    final_answer: str
    sources: list[dict[str, Any]]

    # 7. Response validation
    validation_result: dict[str, Any]
    retry_count: int

    # 8. Supervisor control
    next_action: str

    # 9. Error handling
    error: str | None

    # 10. Check retrieval completion
    retrieval_complete: bool