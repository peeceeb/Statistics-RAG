from math import log2
from collections.abc import Mapping, Sequence
from typing import Any


def _unique_in_order(values: Sequence[str]) -> list[str]:
    """Remove duplicate chunk IDs while preserving their ranking order."""

    seen = set()
    result = []

    for value in values:
        if value is None:
            continue

        chunk_id = str(value)

        if not chunk_id or chunk_id in seen:
            continue

        seen.add(chunk_id)
        result.append(chunk_id)

    return result


def _normalize_relevant_chunks(
    relevant_chunks: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    """
    Convert the dataset's relevant_chunks into:
        {chunk_id: relevance_grade}

    If a chunk appears more than once, retain its highest relevance grade.
    """

    relevance_by_id = {}

    for chunk in relevant_chunks:
        chunk_id = chunk.get("chunk_id")
        relevance = chunk.get("relevance", 1)

        if not chunk_id:
            continue

        try:
            relevance = int(relevance)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid relevance grade for chunk {chunk_id!r}"
            ) from exc

        if relevance < 1:
            raise ValueError(
                f"Relevance must be at least 1: {chunk_id!r}"
            )

        chunk_id = str(chunk_id)

        relevance_by_id[chunk_id] = max(
            relevance_by_id.get(chunk_id, 0),
            relevance
        )

    if not relevance_by_id:
        raise ValueError("No relevant chunk IDs were supplied.")

    return relevance_by_id


def _dcg(relevance_scores: Sequence[int]) -> float:
    """Calculate Discounted Cumulative Gain."""

    return sum(
        (2 ** relevance - 1) / log2(rank + 1)
        for rank, relevance in enumerate(relevance_scores, start=1)
    )


def calculate_retrieval_metrics(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunks: Sequence[Mapping[str, Any]],
    k: int
) -> dict[str, Any]:
    """
    Calculate retrieval metrics for one question at cutoff K.

    Precision@K:
        Relevant results in top K / K

    Recall@K:
        Relevant results in top K / total expected relevant chunks

    Reciprocal Rank:
        1 / rank of the first relevant result, or 0 if not found in top K

    nDCG@K:
        DCG@K / ideal DCG@K, using graded relevance.
    """

    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer.")

    relevance_by_id = _normalize_relevant_chunks(relevant_chunks)

    ranked_ids = _unique_in_order(retrieved_chunk_ids)
    top_k_ids = ranked_ids[:k]

    relevance_scores = [
        relevance_by_id.get(chunk_id, 0)
        for chunk_id in top_k_ids
    ]

    relevant_retrieved = sum(
        1 for score in relevance_scores if score > 0
    )

    relevant_total = len(relevance_by_id)

    precision_at_k = relevant_retrieved / k
    recall_at_k = relevant_retrieved / relevant_total

    first_relevant_rank = next(
        (
            rank
            for rank, score in enumerate(relevance_scores, start=1)
            if score > 0
        ),
        None
    )

    reciprocal_rank = (
        1 / first_relevant_rank
        if first_relevant_rank is not None
        else 0.0
    )

    dcg_at_k = _dcg(relevance_scores)

    ideal_scores = sorted(
        relevance_by_id.values(),
        reverse=True
    )[:k]

    ideal_dcg_at_k = _dcg(ideal_scores)

    ndcg_at_k = (
        dcg_at_k / ideal_dcg_at_k
        if ideal_dcg_at_k > 0
        else 0.0
    )

    return {
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "reciprocal_rank": reciprocal_rank,
        "ndcg_at_k": ndcg_at_k,
        "relevant_retrieved": relevant_retrieved,
        "relevant_total": relevant_total,
        "first_relevant_rank": first_relevant_rank,
        "retrieved_count": len(top_k_ids)
    }
