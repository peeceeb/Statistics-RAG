
from backend.agents.state import AgentState


MAX_RETRIES = 2

TASK_ROUTES = {
    "analysis": "analysis",
    "comparison": "comparison",
    "summarization": "summarization",
}


def supervisor_decision(state: AgentState) -> dict:
    """
    Decide the next action based on the current AgentState.
    """

    # 1. Stop if an execution error occurred.
        # 1. Stop if an execution error occurred.
    if state.get("error"):
        return {
            "next_action": "finish",
            "final_answer": (
                "I couldn't complete your request because "
                "an execution error occurred."
            )
        }

    # 2. Handle validation results.
    validation_result = state.get("validation_result")

    if validation_result:
        # The response passed validation.
        if validation_result.get("is_valid"):
            return {"next_action": "finish"}

        # The response failed validation.
        retry_count = state.get("retry_count", 0)

        # Retry if the maximum retry limit is not reached.
        if retry_count < MAX_RETRIES:
            return {
                "next_action": "retry",
                "retry_count": retry_count + 1
            }

        # Stop when the retry limit is reached.
                # Stop when the retry limit is reached.
        return {
            "next_action": "finish",
            "final_answer": (
                "I couldn't generate an answer that passed "
                "evidence validation after two attempts. "
                "Please refine your question or review the "
                "available source documents."
            )
        }

    # 3. Start planning if required.
		   
    if state.get("requires_planning") and not state.get("execution_plan"):
        return {"next_action": "plan"}

    # 4. Retrieve documents before specialist execution.
    if not state.get("retrieval_complete", False):
        return {"next_action": "retrieve"}

    # 5. Execute planned tasks one by one.
    execution_plan = state.get("execution_plan", [])

    if execution_plan:
        current_index = state.get("current_task_index", 0)

										  
        if current_index >= len(execution_plan):
            return {"next_action": "generate_response"}

							   
        current_task = execution_plan[current_index]

        task_type = current_task.get("type", "").lower()

        next_action = TASK_ROUTES.get(task_type)

        if not next_action:
            raise ValueError(
                f"Unsupported task type in execution plan: {task_type}"
            )

        return {
            "current_task": current_task,
            "next_action": next_action,
        }

    # 6. Route simple queries by intent.
    intent = state.get("intent", "general_qa").lower()

    if intent in TASK_ROUTES:
        return {"next_action": TASK_ROUTES[intent]}

    return {"next_action": "generate_response"}
