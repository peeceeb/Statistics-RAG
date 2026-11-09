from langfuse import observe
from backend.services.retrieval_service import langfuse

from flashrank import Ranker, RerankRequest
from langfuse import get_client, observe

FINAL_TOP_K = 5
RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"

# Initialize FlashRank once
ranker = Ranker(
    model_name=RERANKER_MODEL
)


@observe(name="flashrank-reranking")
def rerank_documents(user_query: str, search_results: list) -> list:

    with langfuse.start_as_current_observation(
        as_type="span",
        name="flashrank-reranking",
        input={
            "query": user_query,
            "candidate_count": len(search_results),
            "reranker_model": RERANKER_MODEL
        }
    ) as rerank_observation:

        # Convert LangChain Documents into FlashRank passages
        passages = [
            {
                "id": index,
                "text": doc.page_content,
                "meta": doc.metadata
            }
            for index, doc in enumerate(search_results)
        ]

        # Create reranking request
        rerank_request = RerankRequest(
            query=user_query,
            passages=passages
        )

        # Rank documents by relevance
        reranked_results = ranker.rerank(rerank_request)

        # Keep the top 5 results
        final_results = reranked_results[:FINAL_TOP_K]

        rerank_observation.update(
            output={
                "reranked_count": len(reranked_results),
                "final_count": len(final_results),
                "results": [
                    {
                        "rerank_score": float(result["score"]),
                        "source": result["meta"].get("source"),
                        "page": result["meta"].get("page")
                    }
                    for result in final_results
                ]
            }
        )

    return final_results
