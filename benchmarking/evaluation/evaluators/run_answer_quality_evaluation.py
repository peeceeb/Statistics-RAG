
from pathlib import Path
import sys
import json
import time
from datetime import datetime
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

from langchain_qdrant import (
    QdrantVectorStore,
    FastEmbedSparse,
    RetrievalMode
)
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

from backend.config.settings import CHAT_MODEL
from backend.services.llm_service import get_openai_client
from backend.services.reranker_service import rerank_documents
from backend.agents.nodes.response_generation import (
    response_generation_node
)

# Evaluation history tracker
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

DATASET_PATH = (
    PROJECT_ROOT
    / "benchmarking/evaluation/datasets/retrieval_eval_v3.jsonl"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "benchmarking/evaluation/reports"
)

JUDGE_MODEL = CHAT_MODEL


# --------------------------------------------------
# JUDGE OUTPUT SCHEMA
# --------------------------------------------------

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correctness_score": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5
        },
        "relevance_score": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5
        },
        "completeness_score": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5
        },
        "faithfulness_score": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5
        },
        "expected_facts_assessment": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string"},
                    "addressed": {"type": "boolean"},
                    "supported_by_evidence": {"type": "boolean"},
                    "explanation": {"type": "string"}
                },
                "required": [
                    "fact",
                    "addressed",
                    "supported_by_evidence",
                    "explanation"
                ],
                "additionalProperties": False
            }
        },
        "unsupported_claims": {
            "type": "array",
            "items": {"type": "string"}
        },
        "reasoning": {
            "type": "string"
        }
    },
    "required": [
        "correctness_score",
        "relevance_score",
        "completeness_score",
        "faithfulness_score",
        "expected_facts_assessment",
        "unsupported_claims",
        "reasoning"
    ],
    "additionalProperties": False
}


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

def load_dataset():

    records = []

    with DATASET_PATH.open("r", encoding="utf-8") as file:

        for line in file:

            if line.strip():
                records.append(json.loads(line))

    return records


# --------------------------------------------------
# INITIALIZE RETRIEVAL
# --------------------------------------------------

def initialize_retrieval():

    print("Initializing evaluation retrieval...")

    client = QdrantClient(
        url=QDRANT_URL,
        timeout=10
    )

    collection_info = client.get_collection(
        COLLECTION_NAME
    )

    configured_dense = (
        collection_info.config.params.vectors
    )

    configured_sparse = (
        collection_info.config.params.sparse_vectors or {}
    )

    dense_vector_name = (
        "dense"
        if isinstance(configured_dense, dict)
        and "dense" in configured_dense
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

        retrieval_method = "Dense + BM25 + RRF"

    else:

        vector_store_options["retrieval_mode"] = (
            RetrievalMode.DENSE
        )

        retrieval_method = "Dense only"

    vector_db = QdrantVectorStore(
        **vector_store_options
    )

    print("Collection:", COLLECTION_NAME)
    print("Points:", collection_info.points_count)
    print("Retrieval method:", retrieval_method)
    print("Retrieval Top K:", RETRIEVAL_TOP_K)
    print("Final Top K:", FINAL_TOP_K)
    print("Generation model:", CHAT_MODEL)
    print("Judge model:", JUDGE_MODEL)

    return vector_db, client, retrieval_method, collection_info


# --------------------------------------------------
# GENERATE ANSWER USING EXISTING NODE
# --------------------------------------------------

def generate_answer(question, reranked_chunks):

    initial_state = {
        "user_query": question,
        "chat_history": [],
        "completed_tasks": [],
        "reranked_chunks": reranked_chunks,
        "validation_result": {},
        "retry_count": 0
    }

    result = response_generation_node(initial_state)

    if result.get("error"):
        raise RuntimeError(result["error"])

    answer = result.get("generated_answer", "").strip()

    if not answer:
        raise RuntimeError(
            "Response generation returned an empty answer."
        )

    return answer


# --------------------------------------------------
# EVALUATE GENERATED ANSWER
# --------------------------------------------------

def evaluate_answer(
    question,
    expected_answer,
    expected_facts,
    generated_answer,
    reranked_chunks
):

    evidence = []

    for index, chunk in enumerate(
        reranked_chunks,
        start=1
    ):

        metadata = chunk.get("meta") or {}

        evidence.append({
            "source_number": index,
            "source": metadata.get("source", "Unknown"),
            "page": metadata.get("page", "Unknown"),
            "chunk_id": metadata.get("chunk_id"),
            "content": chunk.get("text", "")
        })

    judge_instructions = """
You are an independent evaluator of an enterprise RAG system.

Evaluate the generated answer against the question,
reference answer, expected facts, and retrieved evidence.

Use only the supplied information.

SCORING:

1. Correctness:
   Does the answer agree with the reference answer?
   Score from 1 (incorrect) to 5 (fully correct).

2. Relevance:
   Does the answer directly address the question?
   Score from 1 (irrelevant) to 5 (directly relevant).

3. Completeness:
   Does the answer cover the important details required?
   Score from 1 (very incomplete) to 5 (complete).

4. Faithfulness:
   Are the answer's factual claims supported by retrieved evidence?
   Score from 1 (mostly unsupported) to 5 (fully supported).

5. Expected fact assessment:
   Assess EVERY expected fact separately.
   - addressed: Does the answer communicate this fact?
   - supported_by_evidence: Does the retrieved evidence support it?
   - explanation: Briefly explain your decision.

6. Unsupported claims:
   List factual claims in the generated answer that are
   not supported by the retrieved evidence.

IMPORTANT:
- Do not reward an answer merely because it resembles the
  reference answer.
- Check the retrieved evidence independently.
- Do not treat the reference answer as evidence.
- Accept semantically equivalent wording.
- Preserve numerical precision and comparison operators.
- Do not invent missing evidence.
- If evidence is insufficient, reflect that in the scores.
"""

    evaluation_input = {
        "question": question,
        "reference_answer": expected_answer,
        "expected_facts": expected_facts,
        "generated_answer": generated_answer,
        "retrieved_evidence": evidence
    }

    client = get_openai_client()

    response = client.responses.create(
        model=JUDGE_MODEL,
        instructions=judge_instructions,
        input=json.dumps(
            evaluation_input,
            ensure_ascii=False
        ),
        text={
            "format": {
                "type": "json_schema",
                "name": "answer_quality_evaluation",
                "strict": True,
                "schema": JUDGE_SCHEMA
            }
        },
        max_output_tokens=4000
    )

    evaluation = json.loads(response.output_text)

    assessments = evaluation["expected_facts_assessment"]

    if len(assessments) != len(expected_facts):
        raise ValueError(
            "Judge returned an unexpected number of fact assessments."
        )

    addressed_count = sum(
        1 for item in assessments
        if item["addressed"]
    )

    supported_count = sum(
        1 for item in assessments
        if item["supported_by_evidence"]
    )

    total_facts = len(expected_facts)

    evaluation["fact_coverage"] = round(
        addressed_count / total_facts,
        4
    ) if total_facts else 0.0

    evaluation["fact_evidence_support"] = round(
        supported_count / total_facts,
        4
    ) if total_facts else 0.0

    return evaluation, evidence


# --------------------------------------------------
# RUN ONE QUESTION
# --------------------------------------------------

def evaluate_question(
    index,
    total,
    record,
    vector_db
):

    question_id = record["id"]
    question = record["question"]

    print(
        f"\n[{index}/{total}] {question_id}"
    )
    print("Question:", question)

    timings = {}

    question_start = time.perf_counter()

    # 1. Retrieval
    start = time.perf_counter()

    retrieved_documents = vector_db.similarity_search(
        query=question,
        k=RETRIEVAL_TOP_K
    )

    timings["retrieval_seconds"] = round(
        time.perf_counter() - start, 4
    )

    # 2. Reranking
    start = time.perf_counter()

    reranked_chunks = rerank_documents(
        question,
        retrieved_documents
    )

    timings["reranking_seconds"] = round(
        time.perf_counter() - start, 4
    )

    # 3. Answer generation
    start = time.perf_counter()

    generated_answer = generate_answer(
        question,
        reranked_chunks
    )

    timings["generation_seconds"] = round(
        time.perf_counter() - start, 4
    )

    # 4. Answer-quality evaluation
    start = time.perf_counter()

    evaluation, evidence = evaluate_answer(
        question=question,
        expected_answer=record["expected_answer"],
        expected_facts=record["expected_facts"],
        generated_answer=generated_answer,
        reranked_chunks=reranked_chunks
    )

    timings["judging_seconds"] = round(
        time.perf_counter() - start, 4
    )

    timings["total_seconds"] = round(
        time.perf_counter() - question_start, 4
    )

    result = {
        "id": question_id,
        "category": record.get("category"),
        "question": question,
        "expected_answer": record["expected_answer"],
        "expected_facts": record["expected_facts"],
        "generated_answer": generated_answer,
        "retrieved_evidence": evidence,
        "evaluation": evaluation,
        "timings": timings,
        "status": "completed"
    }

    print(
        "Correctness:",
        evaluation["correctness_score"],
        "| Faithfulness:",
        evaluation["faithfulness_score"],
        "| Fact coverage:",
        evaluation["fact_coverage"]
    )

    return result


# --------------------------------------------------
# AGGREGATE RESULTS
# --------------------------------------------------

def calculate_summary(results):

    completed = [
        item for item in results
        if item["status"] == "completed"
    ]

    if not completed:
        return {
            "completed_questions": 0,
            "failed_questions": len(results)
        }

    score_names = [
        "correctness_score",
        "relevance_score",
        "completeness_score",
        "faithfulness_score"
    ]

    summary = {
        "completed_questions": len(completed),
        "failed_questions": len(results) - len(completed)
    }

    for score_name in score_names:

        summary[score_name] = round(
            mean(
                item["evaluation"][score_name]
                for item in completed
            ),
            4
        )

    summary["fact_coverage"] = round(
        mean(
            item["evaluation"]["fact_coverage"]
            for item in completed
        ),
        4
    )

    summary["fact_evidence_support"] = round(
        mean(
            item["evaluation"]["fact_evidence_support"]
            for item in completed
        ),
        4
    )

    return summary


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 70)
    print("ANSWER QUALITY EVALUATION")
    print("=" * 70)

    records = load_dataset()

    print("Dataset:", DATASET_PATH)
    print("Questions:", len(records))

    vector_db, client, retrieval_method, collection_info = (
        initialize_retrieval()
    )

    results = []

    total_start = time.perf_counter()

    for index, record in enumerate(records, start=1):

        try:

            result = evaluate_question(
                index=index,
                total=len(records),
                record=record,
                vector_db=vector_db
            )

        except Exception as exc:

            print(
                f"ERROR evaluating {record.get('id')}: {exc}"
            )

            result = {
                "id": record.get("id"),
                "question": record.get("question"),
                "status": "failed",
                "error": str(exc)
            }

        results.append(result)

    total_duration = time.perf_counter() - total_start

    summary = calculate_summary(results)

    # Save report
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    report_path = REPORT_DIR / (
        f"answer_quality_evaluation_{timestamp}.json"
    )

    report = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "evaluation_type": "answer_quality",
        "collection": COLLECTION_NAME,
        "collection_points": collection_info.points_count,
        "dataset": str(DATASET_PATH),
        "total_questions": len(records),
        "retrieval_method": retrieval_method,
        "embedding_model": EMBEDDING_MODEL,
        "sparse_model": SPARSE_MODEL,
        "reranker_model": "ms-marco-MiniLM-L-12-v2",
        "generation_model": CHAT_MODEL,
        "judge_model": JUDGE_MODEL,
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "final_top_k": FINAL_TOP_K,
        "summary": summary,
        "total_duration_seconds": round(
            total_duration, 2
        ),
        "results": results
    }

    with report_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Update evaluation history.
    # A CSV failure must not invalidate the JSON report.
    try:

        record_evaluation(report_path)

    except Exception as exc:

        print(
            f"WARNING: Could not update evaluation history: {exc}"
        )

    # Final summary
    print("\n" + "=" * 70)
    print("ANSWER QUALITY EVALUATION SUMMARY")
    print("=" * 70)

    print("Questions:", len(records))
    print("Completed:", summary["completed_questions"])
    print("Failed:", summary["failed_questions"])

    for metric, value in summary.items():

        if metric not in (
            "completed_questions",
            "failed_questions"
        ):
            print(f"{metric}: {value}")

    print(
        "\nTotal duration:",
        round(total_duration, 2),
        "seconds"
    )

    print("\nReport saved to:", report_path)
    print("=" * 70)


if __name__ == "__main__":
    main()
