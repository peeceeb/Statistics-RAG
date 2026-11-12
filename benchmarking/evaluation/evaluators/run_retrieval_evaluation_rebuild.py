
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

# Load environment variables before importing backend services.
load_dotenv(PROJECT_ROOT / ".env")

import json
import time
from datetime import datetime
from statistics import mean

from langchain_qdrant import (
    QdrantVectorStore,
    FastEmbedSparse,
    RetrievalMode
)
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

from backend.services.reranker_service import rerank_documents

from benchmarking.evaluation.evaluators.retrieval_metrics import (
    calculate_retrieval_metrics
)

from benchmarking.evaluation.evaluators.evaluation_history import (
    record_evaluation
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

COLLECTION_NAME = "Internal_Docs_Rebuild"
QDRANT_URL = "http://localhost:6333"

RETRIEVAL_TOP_K = 20
FINAL_TOP_K = 5

EMBEDDING_MODEL = "text-embedding-3-large"
SPARSE_MODEL = "Qdrant/bm25"

DATASET_PATH = Path(
    "benchmarking/evaluation/datasets/retrieval_eval_v1.jsonl"
)

REPORT_DIR = Path(
    "benchmarking/evaluation/reports"
)


# --------------------------------------------------
# INITIALIZE QDRANT AND EMBEDDINGS
# --------------------------------------------------

print("Initializing rebuild collection benchmark...")

client = QdrantClient(
    url=QDRANT_URL,
    timeout=10
)

collection_info = client.get_collection(COLLECTION_NAME)

configured_dense = collection_info.config.params.vectors
configured_sparse = collection_info.config.params.sparse_vectors or {}

dense_vector_name = (
    "dense"
    if isinstance(configured_dense, dict) and "dense" in configured_dense
    else ""
)

embedding_model = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    request_timeout=15,
    max_retries=0
)

vector_store_options = {
    "client": client,
    "collection_name": COLLECTION_NAME,
    "embedding": embedding_model,
    "vector_name": dense_vector_name
}

if configured_sparse:

    sparse_vector_name = (
        "sparse"
        if "sparse" in configured_sparse
        else next(iter(configured_sparse))
    )

    sparse_embedding_model = FastEmbedSparse(
        model_name=SPARSE_MODEL
    )

    vector_store_options.update({
        "sparse_embedding": sparse_embedding_model,
        "retrieval_mode": RetrievalMode.HYBRID,
        "sparse_vector_name": sparse_vector_name
    })

    RETRIEVAL_METHOD = "Dense + BM25 + RRF"

else:

    vector_store_options["retrieval_mode"] = RetrievalMode.DENSE

    RETRIEVAL_METHOD = "Dense only"

vector_db = QdrantVectorStore(**vector_store_options)

print("Collection:", COLLECTION_NAME)
print("Points:", collection_info.points_count)
print("Retrieval method:", RETRIEVAL_METHOD)
print("Initial Top K:", RETRIEVAL_TOP_K)
print("Final Top K:", FINAL_TOP_K)


# --------------------------------------------------
# LOAD GOLDEN DATASET
# --------------------------------------------------

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    evaluation_records = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

print("\nEvaluation questions:", len(evaluation_records))


# --------------------------------------------------
# RUN EVALUATION
# --------------------------------------------------

results = []

initial_metrics_list = []
reranked_metrics_list = []

total_start = time.perf_counter()

for index, record in enumerate(evaluation_records, start=1):

    question_id = record["id"]
    query = record["question"]
    relevant_chunks = record["relevant_chunks"]

    print(f"\n[{index}/{len(evaluation_records)}] {question_id}")
    print("Question:", query)

    # Initial retrieval
    retrieval_start = time.perf_counter()

    retrieved_documents = vector_db.similarity_search(
        query=query,
        k=RETRIEVAL_TOP_K
    )

    retrieval_latency = time.perf_counter() - retrieval_start

    initial_ids = [
        doc.metadata.get("chunk_id")
        for doc in retrieved_documents
        if doc.metadata.get("chunk_id")
    ]

    # Initial retrieval metrics
    initial_metrics = calculate_retrieval_metrics(
        retrieved_chunk_ids=initial_ids,
        relevant_chunks=relevant_chunks,
        k=RETRIEVAL_TOP_K
    )

    # Reranking
    rerank_start = time.perf_counter()

    reranked_results = rerank_documents(
        query,
        retrieved_documents
    )

    rerank_latency = time.perf_counter() - rerank_start

    reranked_ids = [
        item["meta"].get("chunk_id")
        for item in reranked_results
        if item.get("meta", {}).get("chunk_id")
    ]

    # Reranked metrics
    reranked_metrics = calculate_retrieval_metrics(
        retrieved_chunk_ids=reranked_ids,
        relevant_chunks=relevant_chunks,
        k=FINAL_TOP_K
    )

    initial_metrics_list.append(initial_metrics)
    reranked_metrics_list.append(reranked_metrics)

    result = {
        "id": question_id,
        "question": query,
        "expected_chunks": relevant_chunks,
        "initial_retrieved_chunk_ids": initial_ids,
        "reranked_chunk_ids": reranked_ids,
        "initial_metrics": initial_metrics,
        "reranked_metrics": reranked_metrics,
        "retrieval_latency_seconds": round(retrieval_latency, 4),
        "reranking_latency_seconds": round(rerank_latency, 4)
    }

    results.append(result)

    print(
        "Initial Recall:",
        initial_metrics["recall_at_k"],
        "| Reranked Recall:",
        reranked_metrics["recall_at_k"]
    )


# --------------------------------------------------
# AGGREGATE METRICS
# --------------------------------------------------

metric_names = [
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "ndcg_at_k"
]

initial_summary = {
    metric: round(
        mean(item[metric] for item in initial_metrics_list),
        4
    )
    for metric in metric_names
}

reranked_summary = {
    metric: round(
        mean(item[metric] for item in reranked_metrics_list),
        4
    )
    for metric in metric_names
}

total_duration = time.perf_counter() - total_start

average_retrieval_latency = mean(
    item["retrieval_latency_seconds"]
    for item in results
)

average_reranking_latency = mean(
    item["reranking_latency_seconds"]
    for item in results
)


# --------------------------------------------------
# SAVE REPORT
# --------------------------------------------------

REPORT_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

report_path = REPORT_DIR / (
    f"retrieval_evaluation_rebuild_{timestamp}.json"
)

report = {
    "evaluation_timestamp": datetime.now().isoformat(),
    "collection": COLLECTION_NAME,
    "dataset": str(DATASET_PATH),
    "total_questions": len(evaluation_records),
    "retrieval_method": RETRIEVAL_METHOD,
    "embedding_model": EMBEDDING_MODEL,
    "sparse_model": SPARSE_MODEL,
    "reranker_model": "ms-marco-MiniLM-L-12-v2",
    "retrieval_top_k": RETRIEVAL_TOP_K,
    "final_top_k": FINAL_TOP_K,
    "initial_retrieval_metrics": initial_summary,
    "reranked_metrics": reranked_summary,
    "average_retrieval_latency_seconds": round(
        average_retrieval_latency, 4
    ),
    "average_reranking_latency_seconds": round(
        average_reranking_latency, 4
    ),
    "total_duration_seconds": round(total_duration, 2),
    "results": results
}

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4, ensure_ascii=False)


# Update evaluation history without affecting the benchmark result.
try:
    record_evaluation(report_path)
except Exception as exc:
    print(f"WARNING: Could not update evaluation history: {exc}")


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 70)
print("REBUILD COLLECTION EVALUATION SUMMARY")
print("=" * 70)

print("Collection:", COLLECTION_NAME)
print("Questions evaluated:", len(evaluation_records))

print("\nINITIAL RETRIEVAL - TOP 20")
for metric, value in initial_summary.items():
    print(f"{metric}: {value}")

print("\nAFTER RERANKING - TOP 5")
for metric, value in reranked_summary.items():
    print(f"{metric}: {value}")

print("\nAverage retrieval latency:",
      round(average_retrieval_latency, 3), "seconds")

print("Average reranking latency:",
      round(average_reranking_latency, 3), "seconds")

print("Total duration:",
      round(total_duration, 2), "seconds")

print("\nReport saved to:", report_path)
print("=" * 70)
