
import logging
import time

from flashrank import Ranker, RerankRequest
from langfuse import get_client


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

logger = logging.getLogger(__name__)

FINAL_TOP_K = 5

RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"

# Initialize Langfuse client
langfuse = get_client()

# Load the reranking model once when the application starts
ranker = Ranker(
    model_name=RERANKER_MODEL
)


# --------------------------------------------------
# 2. Reranking service
# --------------------------------------------------

def rerank_documents(
    user_query: str,
    search_results: list
) -> list:

    """
    Rerank retrieved documents using FlashRank.

    Measures:
        1. Passage preparation
        2. Reranker model inference
        3. Result processing

    Returns:
        Top 5 reranked documents.
    """

    total_start_time = time.perf_counter()

    # --------------------------------------------------
    # 3. Parent observation: Complete reranking operation
    # --------------------------------------------------

    with langfuse.start_as_current_observation(
        as_type="span",
        name="flashrank-reranking",
        input={
            "query": user_query,
            "candidate_count": len(search_results),
            "reranker_model": RERANKER_MODEL
        }
    ) as rerank_observation:

        try:

            # --------------------------------------------------
            # 4. Passage preparation
            # --------------------------------------------------

            preparation_start = time.perf_counter()

            with langfuse.start_as_current_observation(
                as_type="span",
                name="passage-preparation",
                input={
                    "candidate_count": len(search_results)
                }
            ) as preparation_observation:

                passages = [
                    {
                        "id": index,
                        "text": doc.page_content,
                        "meta": doc.metadata
                    }
                    for index, doc in enumerate(search_results)
                ]

                preparation_duration = (
                    time.perf_counter() - preparation_start
                )

                preparation_observation.update(
                    output={
                        "passage_count": len(passages),
                        "duration_seconds": round(
                            preparation_duration, 3
                        )
                    }
                )

            # --------------------------------------------------
            # 5. Prepare FlashRank request
            # --------------------------------------------------

            rerank_request = RerankRequest(
                query=user_query,
                passages=passages
            )

            # --------------------------------------------------
            # 6. Actual reranker model inference
            # --------------------------------------------------

            inference_start = time.perf_counter()

            with langfuse.start_as_current_observation(
                as_type="span",
                name="reranker-inference",
                input={
                    "reranker_model": RERANKER_MODEL,
                    "candidate_count": len(passages)
                }
            ) as inference_observation:

                reranked_results = ranker.rerank(
                    rerank_request
                )

                inference_duration = (
                    time.perf_counter() - inference_start
                )

                inference_observation.update(
                    output={
                        "reranked_count": len(reranked_results),
                        "duration_seconds": round(
                            inference_duration, 3
                        )
                    }
                )

            # --------------------------------------------------
            # 7. Result processing
            # --------------------------------------------------

            processing_start = time.perf_counter()

            with langfuse.start_as_current_observation(
                as_type="span",
                name="result-processing",
                input={
                    "reranked_count": len(reranked_results),
                    "final_top_k": FINAL_TOP_K
                }
            ) as processing_observation:

                # Keep only the five highest-scoring results
                final_results = reranked_results[:FINAL_TOP_K]

                # Prepare lightweight result metadata for tracing
                result_summary = [
                    {
                        "rerank_score": float(result["score"]),
                        "source": (result.get("meta") or {}).get(
                            "source"
                        ),
                        "page": (result.get("meta") or {}).get(
                            "page"
                        )
                    }
                    for result in final_results
                ]

                processing_duration = (
                    time.perf_counter() - processing_start
                )

                processing_observation.update(
                    output={
                        "final_count": len(final_results),
                        "results": result_summary,
                        "duration_seconds": round(
                            processing_duration, 3
                        )
                    }
                )

            # --------------------------------------------------
            # 8. Update parent observation
            # --------------------------------------------------

            total_duration = (
                time.perf_counter() - total_start_time
            )

            rerank_observation.update(
                output={
                    "status": "completed",
                    "candidate_count": len(search_results),
                    "reranked_count": len(reranked_results),
                    "final_count": len(final_results),
                    "preparation_duration_seconds": round(
                        preparation_duration, 3
                    ),
                    "inference_duration_seconds": round(
                        inference_duration, 3
                    ),
                    "processing_duration_seconds": round(
                        processing_duration, 3
                    ),
                    "total_duration_seconds": round(
                        total_duration, 3
                    )
                }
            )

            logger.info(
                "FlashRank completed | "
                "Total: %.2fs | "
                "Preparation: %.2fs | "
                "Inference: %.2fs | "
                "Processing: %.2fs",
                total_duration,
                preparation_duration,
                inference_duration,
                processing_duration
            )

            return final_results

        except Exception as exc:

            total_duration = (
                time.perf_counter() - total_start_time
            )

            logger.exception(
                "FlashRank failed | Duration: %.2f seconds",
                total_duration
            )

            rerank_observation.update(
                output={
                    "status": "error",
                    "duration_seconds": round(total_duration, 3)
                },
                level="ERROR",
                status_message=str(exc)
            )

            raise
