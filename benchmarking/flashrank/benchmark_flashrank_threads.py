
import copy
import logging
import statistics
import sys
import time
from pathlib import Path

import onnxruntime as ort
from dotenv import load_dotenv
from flashrank import RerankRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from backend.services.retrieval_service import search_documents
from backend.services.reranker_service import (
    ranker,
    RERANKER_MODEL,
)



# --------------------------------------------------
# 1. Benchmark configuration
# --------------------------------------------------

QUERY = "ERR-503"

WARMUP_RUNS = 1
BENCHMARK_RUNS = 5

logger = logging.getLogger(__name__)

original_session = ranker.session


# --------------------------------------------------
# 2. Prepare candidate documents once
# --------------------------------------------------

def prepare_passages(search_results):
    """Convert LangChain documents into FlashRank passages."""

    return [
        {
            "id": index,
            "text": doc.page_content,
            "meta": doc.metadata
        }
        for index, doc in enumerate(search_results)
    ]


# --------------------------------------------------
# 3. Run one benchmark configuration
# --------------------------------------------------

def benchmark_configuration(
    configuration_name,
    session,
    query,
    base_passages
):
    """
    Benchmark FlashRank using a specified ONNX session.

    Measures the complete FlashRank rerank operation,
    including tokenization, inference, scoring and sorting.
    """

    print(f"\n{'=' * 60}")
    print(f"Configuration: {configuration_name}")
    print(f"{'=' * 60}")

    durations = []
    rankings = []

    # Use the supplied ONNX session.
    ranker.session = session

    try:
        # Warm-up runs are excluded from measurements.
        for _ in range(WARMUP_RUNS):
            passages = copy.deepcopy(base_passages)

            ranker.rerank(
                RerankRequest(
                    query=query,
                    passages=passages
                )
            )

        # Actual benchmark runs.
        for run_number in range(1, BENCHMARK_RUNS + 1):

            passages = copy.deepcopy(base_passages)

            start_time = time.perf_counter()

            results = ranker.rerank(
                RerankRequest(
                    query=query,
                    passages=passages
                )
            )

            duration = time.perf_counter() - start_time

            durations.append(duration)

            ranking = [
                result["id"]
                for result in results
            ]

            rankings.append(ranking)

            print(
                f"Run {run_number}: {duration:.3f} seconds"
            )

    finally:
        # Restore the original session.
        ranker.session = original_session

    return {
        "name": configuration_name,
        "durations": durations,
        "average": statistics.mean(durations),
        "minimum": min(durations),
        "maximum": max(durations),
        "median": statistics.median(durations),
        "rankings": rankings
    }


# --------------------------------------------------
# 4. Create custom ONNX sessions
# --------------------------------------------------

def create_onnx_session(
    model_path,
    intra_op_threads,
    inter_op_threads
):
    """Create an ONNX session with controlled thread settings."""

    options = ort.SessionOptions()

    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = inter_op_threads

    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=original_session.get_providers()
    )


# --------------------------------------------------
# 5. Main benchmark execution
# --------------------------------------------------

def main():

    print("\nFlashRank ONNX Threading Benchmark")
    print("----------------------------------")

    print("Model:", RERANKER_MODEL)
    print("Query:", QUERY)
    print("Warm-up runs:", WARMUP_RUNS)
    print("Measured runs:", BENCHMARK_RUNS)

    # Find the ONNX model file.
    model_files = list(
        ranker.model_dir.rglob("*.onnx")
    )

    if len(model_files) != 1:
        raise RuntimeError(
            f"Expected one ONNX model file, found: {model_files}"
        )

    model_path = model_files[0]

    print("Model file:", model_path)

    # Retrieve documents only once.
    print("\nRetrieving candidate documents...")

    search_results = search_documents(QUERY)

    if not search_results:
        raise RuntimeError("No documents retrieved.")

    base_passages = prepare_passages(search_results)

    print("Candidates retrieved:", len(base_passages))

    # --------------------------------------------------
    # Benchmark configurations
    # --------------------------------------------------

    configurations = [
        (
            "Baseline - Existing ONNX defaults",
            original_session
        ),
        (
            "Test A - 1 intra-op / 1 inter-op",
            create_onnx_session(
                model_path,
                intra_op_threads=1,
                inter_op_threads=1
            )
        ),
        (
            "Test B - 2 intra-op / 1 inter-op",
            create_onnx_session(
                model_path,
                intra_op_threads=2,
                inter_op_threads=1
            )
        )
    ]

    results = []

    for name, session in configurations:

        result = benchmark_configuration(
            configuration_name=name,
            session=session,
            query=QUERY,
            base_passages=base_passages
        )

        results.append(result)

    # --------------------------------------------------
    # 6. Compare benchmark results
    # --------------------------------------------------

    baseline_ranking = results[0]["rankings"][0]

    print("\n\nBENCHMARK SUMMARY")
    print("=" * 85)

    print(
        f"{'Configuration':<40}"
        f"{'Average':>12}"
        f"{'Median':>12}"
        f"{'Min':>10}"
        f"{'Max':>10}"
    )

    print("-" * 85)

    for result in results:

        print(
            f"{result['name']:<40}"
            f"{result['average']:>11.3f}s"
            f"{result['median']:>11.3f}s"
            f"{result['minimum']:>9.3f}s"
            f"{result['maximum']:>9.3f}s"
        )

    print("\nRANKING CONSISTENCY")
    print("-" * 60)

    for result in results:

        matching_runs = sum(
            ranking == baseline_ranking
            for ranking in result["rankings"]
        )

        print(
            f"{result['name']}: "
            f"{matching_runs}/{BENCHMARK_RUNS} runs "
            "matched baseline ranking"
        )

    print("\nBenchmark completed.")


if __name__ == "__main__":
    main()
