
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from langchain_qdrant import (
    QdrantVectorStore,
    FastEmbedSparse,
    RetrievalMode
)
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from flashrank import Ranker, RerankRequest


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

load_dotenv()

COLLECTION_NAME = "Internal_Docs_Rebuild"

QUERY = (
    "Section 3.3 multiple regression ANOVA output: "
    "Performance and Runtime predicting oxygen consumption. "
    "Find the Model F Value, Pr > F, and R-Square."
)

FLASHRANK_MODEL = "ms-marco-MiniLM-L-12-v2"

RETRIEVAL_TOP_K = 20

EXPECTED_CHUNKS = {
    "bf1bbdd11a6624ae230dca77ee51d3dc1bcefac611e97809d7d41d91f6332c9f",
    "e35bd0f5bc6f2875ec3a8d280225c4ca00a371e7620071c0cee1c2e5b906758d",
    "74e348f263443fa667b53d7110c52129b7e8a69097b582c244f11cdcf6f885b6"
}

EXACT_ANOVA_CHUNK = (
    "bf1bbdd11a6624ae230dca77ee51d3dc1bcefac611e97809d7d41d91f6332c9f"
)


# --------------------------------------------------
# 2. Helper Functions
# --------------------------------------------------

def get_chunk_id(document):
    return document.metadata.get("chunk_id")


def calculate_metrics(ranked_results):

    ranked_chunk_ids = [
        get_chunk_id(result)
        for result in ranked_results
    ]

    top5_ids = set(ranked_chunk_ids[:5])
    top10_ids = set(ranked_chunk_ids[:10])

    exact_rank = (
        ranked_chunk_ids.index(EXACT_ANOVA_CHUNK) + 1
        if EXACT_ANOVA_CHUNK in ranked_chunk_ids
        else None
    )

    return {
        "top5_expected_found": len(
            top5_ids.intersection(EXPECTED_CHUNKS)
        ),
        "top10_expected_found": len(
            top10_ids.intersection(EXPECTED_CHUNKS)
        ),
        "exact_anova_chunk_rank": exact_rank
    }


def print_results(model_name, results, latency):

    metrics = calculate_metrics(results)

    print("\n" + "=" * 80)
    print(model_name)
    print("=" * 80)

    print("Inference latency:", round(latency, 3), "seconds")
    print(
        "Expected chunks in Top-5:",
        metrics["top5_expected_found"],
        "/ 3"
    )
    print(
        "Expected chunks in Top-10:",
        metrics["top10_expected_found"],
        "/ 3"
    )
    print(
        "Exact ANOVA chunk rank:",
        metrics["exact_anova_chunk_rank"]
    )

    print("\nRanked documents:")

    for rank, result in enumerate(results, start=1):

        chunk_id = get_chunk_id(result)

        print(
            f"{rank:02d}. "
            f"Chunk: {chunk_id} | "
            f"Page: {result.metadata.get('page')} | "
            f"Expected: {chunk_id in EXPECTED_CHUNKS}"
        )

    return metrics


# --------------------------------------------------
# 3. Initialize Qdrant
# --------------------------------------------------

print("\nInitializing Qdrant...")

client = QdrantClient(
    url="http://localhost:6333",
    timeout=10
)

info = client.get_collection(COLLECTION_NAME)

dense_config = info.config.params.vectors
sparse_config = info.config.params.sparse_vectors or {}

dense_name = (
    "dense"
    if isinstance(dense_config, dict) and "dense" in dense_config
    else ""
)


# --------------------------------------------------
# 4. Initialize Embeddings
# --------------------------------------------------

print("\nInitializing embeddings...")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    request_timeout=15,
    max_retries=0
)


# --------------------------------------------------
# 5. Configure Vector Store
# --------------------------------------------------

options = {
    "client": client,
    "collection_name": COLLECTION_NAME,
    "embedding": embeddings,
    "vector_name": dense_name
}

if sparse_config:

    sparse_name = (
        "sparse"
        if "sparse" in sparse_config
        else next(iter(sparse_config))
    )

    options.update({
        "sparse_embedding": FastEmbedSparse(
            model_name="Qdrant/bm25"
        ),
        "retrieval_mode": RetrievalMode.HYBRID,
        "sparse_vector_name": sparse_name
    })

else:

    options["retrieval_mode"] = RetrievalMode.DENSE


vector_db = QdrantVectorStore(**options)


# --------------------------------------------------
# 6. Retrieve Candidates Once
# --------------------------------------------------

print("\nRetrieving candidates...")

retrieval_start = time.perf_counter()

documents = vector_db.similarity_search(
    QUERY,
    k=RETRIEVAL_TOP_K
)

retrieval_latency = time.perf_counter() - retrieval_start

if len(documents) != RETRIEVAL_TOP_K:
    raise RuntimeError(
        f"Expected {RETRIEVAL_TOP_K} candidates, "
        f"received {len(documents)}."
    )

candidate_ids = [get_chunk_id(doc) for doc in documents]

if len(candidate_ids) != len(set(candidate_ids)):
    raise RuntimeError(
        "Duplicate chunk IDs detected. "
        "Evaluation stopped to preserve candidate integrity."
    )

print("Retrieved candidates:", len(documents))
print("Retrieval latency:", round(retrieval_latency, 3), "seconds")

print(
    "Expected chunks in initial retrieval:",
    len(set(candidate_ids).intersection(EXPECTED_CHUNKS)),
    "/ 3"
)


# --------------------------------------------------
# 7. Prepare FlashRank Passages
# --------------------------------------------------

passages = [
    {
        "id": index,
        "text": doc.page_content,
        "meta": doc.metadata
    }
    for index, doc in enumerate(documents)
]


# --------------------------------------------------
# 8. FlashRank Evaluation
# --------------------------------------------------

print("\nLoading FlashRank...")

flashrank = Ranker(
    model_name=FLASHRANK_MODEL
)

flashrank_request = RerankRequest(
    query=QUERY,
    passages=passages
)

print("\nRunning FlashRank inference...")

start = time.perf_counter()

flashrank_output = flashrank.rerank(
    flashrank_request
)

flashrank_latency = time.perf_counter() - start

flashrank_results = [
    documents[int(result["id"])]
    for result in flashrank_output
]

flashrank_metrics = print_results(
    "FLASHRANK RESULTS",
    flashrank_results,
    flashrank_latency
)


# --------------------------------------------------
# 9. Evaluation Summary
# --------------------------------------------------

print("\n" + "=" * 80)
print("FLASHRANK EVALUATION SUMMARY")
print("=" * 80)

comparison = {
    "flashrank": {
        **flashrank_metrics,
        "inference_latency_seconds": round(
            flashrank_latency, 4
        )
    }
}

print(json.dumps(comparison, indent=4))


# --------------------------------------------------
# 10. Save Report
# --------------------------------------------------

report = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "collection": COLLECTION_NAME,
    "query": QUERY,
    "retrieval_top_k": RETRIEVAL_TOP_K,
    "retrieval_latency_seconds": round(
        retrieval_latency, 4
    ),
    "candidate_chunk_ids": candidate_ids,
    "expected_chunk_ids": sorted(EXPECTED_CHUNKS),
    "flashrank_model": FLASHRANK_MODEL,
    "evaluation": comparison
}

report_directory = (
    PROJECT_ROOT
    / "benchmarking"
    / "evaluation"
    / "reports"
)

report_directory.mkdir(
    parents=True,
    exist_ok=True
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

report_path = (
    report_directory
    / f"flashrank_evaluation_{timestamp}.json"
)

with open(report_path, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)

print("\nEvaluation report saved:")
print(report_path)

print("\nFlashRank evaluation completed.")
print("No production data or production code was modified.")
