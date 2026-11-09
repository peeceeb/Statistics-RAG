
import logging
import time

from langgraph.graph import StateGraph, START, END

from backend.agents.state import AgentState
from backend.agents.supervisor import supervisor_decision

from backend.agents.nodes.query_understanding import (
    query_understanding_node
)

from backend.agents.nodes.planning import (
    planning_node
)

from backend.agents.nodes.retrieval import (
    retrieval_node
)

from backend.agents.nodes.analysis import (
    analysis_node
)

from backend.agents.nodes.comparison import (
    comparison_node
)

from backend.agents.nodes.summarization import (
    summarization_node
)

from backend.agents.nodes.response_generation import (
    response_generation_node
)

from backend.agents.nodes.validation import (
    validation_node
)


# --------------------------------------------------
# 1. Configure logging
# --------------------------------------------------

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 2. Reusable node logging wrapper
# --------------------------------------------------

def log_node(node_name: str, node_function):
    """
    Wrap a LangGraph node with execution logging.

    Logs:
        - Node start
        - Node completion
        - Execution duration
        - Execution errors

    Returns:
        Updated AgentState fields from the node.
    """

    def wrapper(state: AgentState) -> dict:

        start_time = time.perf_counter()

        logger.info(
            "Agent node started: %s",
            node_name
        )

        try:

            result = node_function(state)

            duration = time.perf_counter() - start_time

            # Log execution errors returned by the node.
            if result.get("error"):

                logger.error(
                    "Agent node reported an error: %s | Error: %s",
                    node_name,
                    result.get("error")
                )

            else:

                logger.info(
                    "Agent node completed: %s | Duration: %.2f seconds",
                    node_name,
                    duration
                )

            # Log validation results.
            if "validation_result" in result:

                validation_result = result.get(
                    "validation_result",
                    {}
                )

                if validation_result:

                    logger.info(
                        "Validation result: valid=%s",
                        validation_result.get("is_valid")
                    )

            return result

        except Exception:

            duration = time.perf_counter() - start_time

            logger.exception(
                "Agent node failed: %s | Duration: %.2f seconds",
                node_name,
                duration
            )

            raise

    return wrapper


# --------------------------------------------------
# 3. Supervisor routing
# --------------------------------------------------

def route_supervisor(state: AgentState) -> str:
    """
    Route execution based on the Supervisor's decision.
    """

    next_action = state.get("next_action", "finish")

    logger.info(
        "Supervisor decision: %s | Retry count: %s",
        next_action,
        state.get("retry_count", 0)
    )

    return next_action


# --------------------------------------------------
# 4. Specialist routing
# --------------------------------------------------

def route_after_specialist(state: AgentState) -> str:
    """
    Decide whether to execute another planned task
    or generate the final response.
    """

    execution_plan = state.get("execution_plan", [])

    if execution_plan:

        logger.info(
            "Specialist completed | Next: supervisor | Task index: %s",
            state.get("current_task_index", 0)
        )

        return "supervisor"

    logger.info(
        "Specialist completed | Next: response_generation"
    )

    return "response_generation"


# --------------------------------------------------
# 5. Build LangGraph workflow
# --------------------------------------------------

def build_agent_graph():

    workflow = StateGraph(AgentState)

    # Register nodes with logging wrappers.

    workflow.add_node(
        "query_understanding",
        log_node(
            "query_understanding",
            query_understanding_node
        )
    )

    workflow.add_node(
        "supervisor",
        log_node(
            "supervisor",
            supervisor_decision
        )
    )

    workflow.add_node(
        "planning",
        log_node(
            "planning",
            planning_node
        )
    )

    workflow.add_node(
        "retrieval",
        log_node(
            "retrieval",
            retrieval_node
        )
    )

    workflow.add_node(
        "analysis",
        log_node(
            "analysis",
            analysis_node
        )
    )

    workflow.add_node(
        "comparison",
        log_node(
            "comparison",
            comparison_node
        )
    )

    workflow.add_node(
        "summarization",
        log_node(
            "summarization",
            summarization_node
        )
    )

    workflow.add_node(
        "response_generation",
        log_node(
            "response_generation",
            response_generation_node
        )
    )

    workflow.add_node(
        "validation",
        log_node(
            "validation",
            validation_node
        )
    )

    # Initial entry point.

    workflow.add_edge(
        START,
        "query_understanding"
    )

    # Query understanding -> Supervisor.

    workflow.add_edge(
        "query_understanding",
        "supervisor"
    )

    # Supervisor conditional routing.

    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "plan": "planning",
            "retrieve": "retrieval",
            "analysis": "analysis",
            "comparison": "comparison",
            "summarization": "summarization",
            "generate_response": "response_generation",
            "retry": "response_generation",
            "finish": END
        }
    )

    # Planning -> Supervisor.

    workflow.add_edge(
        "planning",
        "supervisor"
    )

    # Retrieval -> Supervisor.

    workflow.add_edge(
        "retrieval",
        "supervisor"
    )

    # Specialist routing.

    workflow.add_conditional_edges(
        "analysis",
        route_after_specialist,
        {
            "supervisor": "supervisor",
            "response_generation": "response_generation"
        }
    )

    workflow.add_conditional_edges(
        "comparison",
        route_after_specialist,
        {
            "supervisor": "supervisor",
            "response_generation": "response_generation"
        }
    )

    workflow.add_conditional_edges(
        "summarization",
        route_after_specialist,
        {
            "supervisor": "supervisor",
            "response_generation": "response_generation"
        }
    )

    # Response generation -> Validation.

    workflow.add_edge(
        "response_generation",
        "validation"
    )

    # Validation -> Supervisor.

    workflow.add_edge(
        "validation",
        "supervisor"
    )

    # Compile workflow.

    logger.info("Compiling Enterprise RAG agent graph.")

    return workflow.compile()


# --------------------------------------------------
# 6. Create reusable graph instance
# --------------------------------------------------

agent_graph = build_agent_graph()

logger.info("Enterprise RAG agent graph initialized.")
