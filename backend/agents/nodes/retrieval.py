
import logging

from backend.agents.state import AgentState
from backend.services.retrieval_service import search_documents
from backend.services.reranker_service import rerank_documents


logger = logging.getLogger(__name__)


def retrieval_node(state: AgentState) -> dict:
    """
    Retrieve and rerank documents for the user's query.

    Reads:
        user_query

    Updates:
        retrieved_documents
        reranked_chunks
        retrieval_complete
        error
    """

    try:
        user_query = state.get("user_query", "").strip()

        if not user_query:
            raise ValueError("User query cannot be empty.")

        # Step 1: Retrieve documents from Internal_Docs.
        retrieved_documents = search_documents(user_query)

        # Step 2: Rerank retrieved documents.
        # Skip reranking when retrieval returns no documents.
        if retrieved_documents:
            reranked_chunks = rerank_documents(
                user_query,
                retrieved_documents
            )
        else:
            reranked_chunks = []

        # Step 3: Return updated state fields.
        return {
            "retrieved_documents": retrieved_documents,
            "reranked_chunks": reranked_chunks,
            "retrieval_complete": True,
            "error": None
        }

    except Exception as exc:

        logger.exception("Document retrieval failed.")

        return {
            "retrieved_documents": [],
            "reranked_chunks": [],
            "retrieval_complete": False,
            "error": str(exc)
        }
