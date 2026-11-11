
def get_unique_reranked_chunks(reranked_chunks: list) -> list:
    """
    Remove duplicate retrieved chunks while preserving their order.

    Rules:
    1. Deduplicate using chunk_id.
    2. Preserve chunks without chunk_id.
    3. Preserve the original reranking order.
    """

    unique_chunks = []
    seen_chunk_ids = set()

    for chunk in reranked_chunks or []:

        metadata = chunk.get("meta") or {}
        chunk_id = metadata.get("chunk_id")

        # Deduplicate only when a valid chunk_id exists.
        if chunk_id is not None and str(chunk_id).strip():

            normalized_chunk_id = str(chunk_id)

            if normalized_chunk_id in seen_chunk_ids:
                continue

            seen_chunk_ids.add(normalized_chunk_id)

        # Retain the chunk, including chunks without chunk_id.
        unique_chunks.append(chunk)

    return unique_chunks
