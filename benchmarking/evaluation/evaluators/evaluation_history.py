
import csv
import json
from pathlib import Path


EVALUATION_DIR = Path(__file__).resolve().parents[1]

HISTORY_FILE = EVALUATION_DIR / "evaluation_history.csv"


COLUMNS = [
    "report_file",
    "run_timestamp",
    "evaluation_type",
    "dataset",
    "collection",
    "retrieval_method",
    "embedding_model",
    "sparse_model",
    "reranker_model",
    "generation_model",
    "judge_model",
    "retrieval_top_k",
    "final_top_k",
    "total_questions",
    "failed_questions",
    "initial_precision",
    "initial_recall",
    "initial_mrr",
    "initial_ndcg",
    "reranked_precision",
    "reranked_recall",
    "reranked_mrr",
    "reranked_ndcg",
    "correctness",
    "relevance",
    "completeness",
    "faithfulness",
    "fact_coverage",
    "fact_evidence_support",
    "avg_retrieval_latency_seconds",
    "avg_reranking_latency_seconds",
    "total_duration_seconds",
]


def record_evaluation(report_path: str | Path) -> None:
    """
    Extract evaluation metrics from a JSON report
    and append one row to evaluation_history.csv.

    Duplicate report filenames are not recorded twice.
    """

    report_path = Path(report_path)

    if not report_path.exists():
        raise FileNotFoundError(
            f"Evaluation report not found: {report_path}"
        )

    with report_path.open(
        "r",
        encoding="utf-8"
    ) as file:
        report = json.load(file)

    evaluation_type = report.get(
        "evaluation_type",
        "retrieval"
    )

    summary = report.get("summary", {})

    initial = report.get(
        "initial_retrieval_metrics",
        {}
    )

    reranked = report.get(
        "reranked_metrics",
        {}
    )

    row = {
        "report_file": report_path.name,
        "run_timestamp": report.get("evaluation_timestamp"),
        "evaluation_type": evaluation_type,
        "dataset": Path(report.get("dataset", "")).name,
        "collection": report.get("collection"),
        "retrieval_method": report.get("retrieval_method"),
        "embedding_model": report.get("embedding_model"),
        "sparse_model": report.get("sparse_model"),
        "reranker_model": report.get("reranker_model"),
        "generation_model": report.get("generation_model"),
        "judge_model": report.get("judge_model"),
        "retrieval_top_k": report.get("retrieval_top_k"),
        "final_top_k": report.get("final_top_k"),
        "total_questions": report.get("total_questions"),
        "failed_questions": summary.get("failed_questions"),
        "initial_precision": initial.get("precision_at_k"),
        "initial_recall": initial.get("recall_at_k"),
        "initial_mrr": initial.get("reciprocal_rank"),
        "initial_ndcg": initial.get("ndcg_at_k"),
        "reranked_precision": reranked.get("precision_at_k"),
        "reranked_recall": reranked.get("recall_at_k"),
        "reranked_mrr": reranked.get("reciprocal_rank"),
        "reranked_ndcg": reranked.get("ndcg_at_k"),
        "correctness": summary.get("correctness_score"),
        "relevance": summary.get("relevance_score"),
        "completeness": summary.get("completeness_score"),
        "faithfulness": summary.get("faithfulness_score"),
        "fact_coverage": summary.get("fact_coverage"),
        "fact_evidence_support": summary.get(
            "fact_evidence_support"
        ),
        "avg_retrieval_latency_seconds": report.get(
            "average_retrieval_latency_seconds"
        ),
        "avg_reranking_latency_seconds": report.get(
            "average_reranking_latency_seconds"
        ),
        "total_duration_seconds": report.get(
            "total_duration_seconds"
        ),
    }

    HISTORY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Prevent duplicate entries for the same report.
    if HISTORY_FILE.exists():
        with HISTORY_FILE.open(
            "r",
            newline="",
            encoding="utf-8-sig"
        ) as file:
            existing_rows = csv.DictReader(file)

            if any(
                existing.get("report_file") == report_path.name
                for existing in existing_rows
            ):
                print(
                    "Evaluation already recorded:",
                    report_path.name
                )
                return

    file_exists = HISTORY_FILE.exists()

    with HISTORY_FILE.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=COLUMNS
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    print(
        "Evaluation history updated:",
        HISTORY_FILE
    )
