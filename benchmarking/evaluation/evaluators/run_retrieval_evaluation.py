import json
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from statistics import mean

from dotenv import load_dotenv

# Resolve project root and load environment configuration
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from backend.services.retrieval_service import (
    search_documents,
    RETRIEVAL_TOP_K,
    RETRIEVAL_METHOD
)
from backend.services.reranker_service import (
    rerank_documents,
    FINAL_TOP_K,
    RERANKER_MODEL
)

from benchmarking.evaluation.evaluators.retrieval_metrics import (
    calculate_retrieval_metrics
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATASET_PATH = (
    PROJECT_ROOT
    / "benchmarking"
    / "evaluation"
    / "datasets"
    / "retrieval_eval_v1.jsonl"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "benchmarking"
    / "evaluation"
    / "reports"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load golden evaluation dataset
# --------------------------------------------------

def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Golden dataset not found: {DATASET_PATH}"
        )

    records = []

    with DATASET_PATH.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from exc

    if not records:
        raise ValueError("Golden dataset is empty.")

    ids = [record["id"] for record in records]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evaluation record IDs found.")

    return records


# --------------------------------------------------
# 3. Extract chunk IDs from retrieval results
# --------------------------------------------------

def extract_chunk_id(result):
    """
    Support both LangChain Documents and FlashRank result dictionaries.
    """

    if isinstance(result, dict):
        metadata = result.get("meta") or result.get("metadata") or {}
    else:
        metadata = getattr(result, "metadata", {}) or {}

    # Support nested metadata as well as flattened metadata.
    chunk_id = metadata.get("chunk_id")

    if not chunk_id and isinstance(metadata.get("metadata"), dict):
        chunk_id = metadata["metadata"].get("chunk_id")

    if not chunk_id:
        raise ValueError(
            "Retrieved result has no chunk_id. "
            f"Available metadata keys: {list(metadata.keys())}"
        )

    return str(chunk_id)


# --------------------------------------------------
# 4. Run retrieval evaluation
# --------------------------------------------------

def evaluate_question(record):

    question_id = record["id"]
    question = record["question"]
    relevant_chunks = record["relevant_chunks"]

    print(f"\nEvaluating {question_id}: {question}")

    # Stage A: Initial Qdrant retrieval
    retrieval_start = time.perf_counter()

    retrieved_documents = search_documents(question)

    retrieval_duration = time.perf_counter() - retrieval_start

    initial_chunk_ids = [
        extract_chunk_id(document)
        for document in retrieved_documents
    ]

    # Stage B: FlashRank reranking
    reranking_start = time.perf_counter()

    if retrieved_documents:
        reranked_results = rerank_documents(
            question,
            retrieved_documents
        )
    else:
        reranked_results = []

    reranking_duration = time.perf_counter() - reranking_start

    reranked_chunk_ids = [
        extract_chunk_id(result)
        for result in reranked_results
    ]

    # Stage C: Calculate metrics for both stages
    initial_metrics = calculate_retrieval_metrics(
        retrieved_chunk_ids=initial_chunk_ids,
        relevant_chunks=relevant_chunks,
        k=RETRIEVAL_TOP_K
    )

    reranked_metrics = calculate_retrieval_metrics(
        retrieved_chunk_ids=reranked_chunk_ids,
        relevant_chunks=relevant_chunks,
        k=FINAL_TOP_K
    )

    print(
        f"  Initial Recall@{RETRIEVAL_TOP_K}: "
        f"{initial_metrics['recall_at_k']:.3f}"
    )

    print(
        f"  Reranked Recall@{FINAL_TOP_K}: "
        f"{reranked_metrics['recall_at_k']:.3f}"
    )

    return {
        "id": question_id,
        "question": question,
        "category": record.get("category"),

        "initial_retrieval": {
            "retrieved_count": len(initial_chunk_ids),
            "retrieved_chunk_ids": initial_chunk_ids,
            "latency_seconds": round(retrieval_duration, 3),
            "metrics": initial_metrics
        },

        "reranked_retrieval": {
            "retrieved_count": len(reranked_chunk_ids),
            "retrieved_chunk_ids": reranked_chunk_ids,
            "latency_seconds": round(reranking_duration, 3),
            "metrics": reranked_metrics
        }
    }


# --------------------------------------------------
# 5. Aggregate evaluation metrics
# --------------------------------------------------

def aggregate_metrics(results, stage):

    metric_names = [
        "precision_at_k",
        "recall_at_k",
        "reciprocal_rank",
        "ndcg_at_k"
    ]

    aggregate = {}

    for metric_name in metric_names:
        values = [
            result[stage]["metrics"][metric_name]
            for result in results
        ]

        aggregate[metric_name] = round(mean(values), 4)

    aggregate["average_latency_seconds"] = round(
        mean([
            result[stage]["latency_seconds"]
            for result in results
        ]),
        3
    )

    return aggregate


# --------------------------------------------------
# 6. Main execution
# --------------------------------------------------

def main():

    records = load_dataset()

    print("\n========== RETRIEVAL EVALUATION ==========")
    print("Dataset:", DATASET_PATH.name)
    print("Evaluation questions:", len(records))
    print("Retrieval method:", RETRIEVAL_METHOD)
    print("Initial retrieval K:", RETRIEVAL_TOP_K)
    print("Final reranking K:", FINAL_TOP_K)
    print("Reranker:", RERANKER_MODEL)

    results = []

    evaluation_start = time.perf_counter()

    for record in records:
        result = evaluate_question(record)
        results.append(result)

    total_duration = time.perf_counter() - evaluation_start

    # Aggregate metrics
    initial_summary = aggregate_metrics(
        results,
        "initial_retrieval"
    )

    reranked_summary = aggregate_metrics(
        results,
        "reranked_retrieval"
    )

    # Build report
    report = {
        "evaluation_name": "Golden Dataset v1 Retrieval Evaluation",
        "evaluation_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "dataset": DATASET_PATH.name,
        "total_questions": len(records),

        "configuration": {
            "collection": "Internal_Docs",
            "retrieval_method": RETRIEVAL_METHOD,
            "retrieval_top_k": RETRIEVAL_TOP_K,
            "reranker": RERANKER_MODEL,
            "final_top_k": FINAL_TOP_K
        },

        "execution": {
            "total_duration_seconds": round(total_duration, 3)
        },

        "initial_retrieval_summary": initial_summary,
        "reranked_retrieval_summary": reranked_summary,

        "question_results": results
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_path = REPORT_DIR / (
        f"retrieval_evaluation_{timestamp}.json"
    )

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    # Display summary
    print("\n========== INITIAL RETRIEVAL SUMMARY ==========")

    for metric, value in initial_summary.items():
        print(f"{metric}: {value}")

    print("\n========== RERANKED RETRIEVAL SUMMARY ==========")

    for metric, value in reranked_summary.items():
        print(f"{metric}: {value}")

    print("\n========== EXECUTION SUMMARY ==========")
    print("Questions evaluated:", len(results))
    print("Total duration:", round(total_duration, 2), "seconds")
    print("Report saved:", report_path)

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
