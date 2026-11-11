
import copy
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from flashrank import Ranker, RerankRequest


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


# --------------------------------------------------
# 2. BENCHMARK CONFIGURATION
# --------------------------------------------------

QUERIES = [
    {
        "name": "Exact keyword",
        "query": "ERR-503",
        "target_keyword": "ERR-503"
    },
    {
        "name": "Conceptual",
        "query": "Explain regression analysis.",
        "target_keyword": None
    },
    {
        "name": "Technical",
        "query": "Explain hybrid retrieval using dense and sparse search.",
        "target_keyword": None
    },
    {
        "name": "Document content",
        "query": (
            "What information is available in the documents "
            "about their purpose and key findings?"
        ),
        "target_keyword": None
    },
    {
        "name": "Comparison",
        "query": "Compare dense retrieval and hybrid retrieval.",
        "target_keyword": None
    }
]

# Run only Document content for this investigation.
# Set to None to run all five queries later.
QUERY_FILTER = "Document content"

CANDIDATE_COUNTS = [20, 15, 10]

WARMUP_RUNS = 1
BENCHMARK_RUNS = 5

FINAL_TOP_K = 5

RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"


# --------------------------------------------------
# 3. INITIALIZE FLASHRANK
# --------------------------------------------------

print("\nInitializing FlashRank...")

ranker = Ranker(model_name=RERANKER_MODEL)

print(f"Model: {RERANKER_MODEL}")


# --------------------------------------------------
# 4. IMPORT EXISTING BACKEND SERVICES
# --------------------------------------------------

from backend.services.retrieval_service import search_documents
from backend.services.source_service import get_unique_reranked_chunks


# --------------------------------------------------
# 5. RETRIEVE AND PREPARE PASSAGES
# --------------------------------------------------

def retrieve_base_passages(query, target_keyword=None):

    print("\nRetrieving documents...")
    print(f"Query: {query}")

    documents = search_documents(query)

    print(f"Documents retrieved: {len(documents)}")

    if not documents:
        return [], set()

    base_passages = []

    for index, doc in enumerate(documents):

        base_passages.append({
            "id": index,
            "text": doc.page_content,
            "meta": doc.metadata
        })

    target_ids = set()

    if target_keyword:

        for passage in base_passages:

            if target_keyword.lower() in passage["text"].lower():
                target_ids.add(passage["id"])

        print(f"Target keyword: {target_keyword}")
        print(f"Matching document IDs: {sorted(target_ids)}")

        if target_ids:
            print(
                "Target retrieval rank:",
                min(target_ids) + 1
            )

    return base_passages, target_ids


# --------------------------------------------------
# 6. INSPECT RETRIEVAL DUPLICATES
# --------------------------------------------------

def inspect_retrieval_duplicates(base_passages):

    groups = defaultdict(list)

    for passage in base_passages:

        metadata = passage.get("meta") or {}
        chunk_id = metadata.get("chunk_id")

        if chunk_id is not None and str(chunk_id).strip():
            groups[str(chunk_id)].append(passage)

    duplicate_groups = {
        chunk_id: passages
        for chunk_id, passages in groups.items()
        if len(passages) > 1
    }

    passages_without_chunk_id = sum(
        1
        for passage in base_passages
        if not (passage.get("meta") or {}).get("chunk_id")
    )

    print("\n" + "=" * 70)
    print("QDRANT RETRIEVAL DUPLICATE INSPECTION")
    print("=" * 70)

    print(f"Total retrieved passages: {len(base_passages)}")
    print(f"Unique chunk IDs: {len(groups)}")
    print(f"Duplicate chunk groups: {len(duplicate_groups)}")
    print(f"Passages without chunk ID: {passages_without_chunk_id}")

    if not duplicate_groups:
        print("\nNo duplicate chunk IDs found.")
        return

    for chunk_id, passages in duplicate_groups.items():

        print("\n" + "-" * 65)
        print(f"Chunk ID: {chunk_id}")
        print(f"Occurrences: {len(passages)}")

        for passage in passages:

            metadata = passage.get("meta") or {}

            print(f"\nPassage ID: {passage['id']}")
            print(f"Source: {metadata.get('source', 'Unknown')}")
            print(f"Qdrant point ID: {metadata.get('_id', 'Unknown')}")
            print(f"Document ID: {metadata.get('document_id', 'Unknown')}")
            print(f"Chunk index: {metadata.get('chunk_index', 'Unknown')}")

            print(
                "Text preview:",
                passage["text"][:150].replace("\n", " ")
            )

        unique_texts = {
            passage["text"]
            for passage in passages
        }

        print(
            "\nIdentical full text:",
            len(unique_texts) == 1
        )


# --------------------------------------------------
# 7. BENCHMARK A CANDIDATE COUNT
# --------------------------------------------------

def benchmark_candidate_count(
    query,
    candidate_count,
    base_passages,
    target_ids
):

    selected_passages = base_passages[:candidate_count]

    print("\n" + "=" * 65)
    print(f"Candidates passed to FlashRank: {candidate_count}")
    print("=" * 65)

    durations = []
    top5_rankings = []
    top5_details = []
    unique_chunk_counts = []

    # Warm-up
    for _ in range(WARMUP_RUNS):

        ranker.rerank(
            RerankRequest(
                query=query,
                passages=copy.deepcopy(selected_passages)
            )
        )

    # Measured runs
    for run_number in range(1, BENCHMARK_RUNS + 1):

        passages = copy.deepcopy(selected_passages)

        start_time = time.perf_counter()

        results = ranker.rerank(
            RerankRequest(
                query=query,
                passages=passages
            )
        )

        duration = time.perf_counter() - start_time

        durations.append(duration)

        top5 = results[:FINAL_TOP_K]

        top5_ids = [
            result["id"]
            for result in top5
        ]

        top5_rankings.append(top5_ids)

        # Count distinct chunk IDs in the top 5.
        top5_chunk_ids = [
            str((result.get("meta") or {}).get("chunk_id"))
            for result in top5
            if (result.get("meta") or {}).get("chunk_id")
        ]

        unique_top5_chunks = len(set(top5_chunk_ids))
        unique_chunk_counts.append(unique_top5_chunks)

        # Capture document details from first measured run.
        if run_number == 1:

            for rank, result in enumerate(top5, start=1):

                metadata = result.get("meta") or {}

                top5_details.append({
                    "rank": rank,
                    "id": result["id"],
                    "source": metadata.get("source", "Unknown"),
                    "page": metadata.get("page", "Unknown"),
                    "chunk_id": metadata.get("chunk_id", "Unknown"),
                    "text_preview": result.get("text", "")[:200]
                })

        target_found = (
            any(result["id"] in target_ids for result in top5)
            if target_ids else None
        )

        target_status = (
            str(target_found)
            if target_found is not None
            else "Not evaluated"
        )

        print(
            f"Run {run_number}: "
            f"{duration:.3f}s | "
            f"Top-5 IDs: {top5_ids} | "
            f"Unique chunks: {unique_top5_chunks} | "
            f"Target in top 5: {target_status}"
        )

    return {
        "candidate_count": candidate_count,
        "average": statistics.mean(durations),
        "median": statistics.median(durations),
        "minimum": min(durations),
        "maximum": max(durations),
        "top5_rankings": top5_rankings,
        "top5_details": top5_details,
        "unique_chunk_counts": unique_chunk_counts
    }


# --------------------------------------------------
# 8. PRINT QUERY SUMMARY
# --------------------------------------------------

def print_query_summary(query_name, query_results):

    if not query_results:
        return

    print("\n\n" + "=" * 100)
    print(f"BENCHMARK SUMMARY: {query_name}")
    print("=" * 100)

    print(
        f"{'Configuration':<25}"
        f"{'Average':<12}"
        f"{'Median':<12}"
        f"{'Min':<12}"
        f"{'Max':<12}"
        f"{'Top-5 overlap':<15}"
        f"{'Unique chunks':<15}"
    )

    # First 20-candidate result is the baseline.
    baseline_ids = set(
        query_results[0]["top5_rankings"][0]
    )

    for result in query_results:

        current_ids = set(
            result["top5_rankings"][0]
        )

        overlap = len(
            baseline_ids.intersection(current_ids)
        )

        # The deduplicated configuration is appended last.
        if result.get("deduplicated"):
            configuration = "Deduplicated"
        else:
            configuration = f"Original {result['candidate_count']}"

        average_unique_chunks = statistics.mean(
            result["unique_chunk_counts"]
        )

        print(
            f"{configuration:<25}"
            f"{result['average']:<12.3f}"
            f"{result['median']:<12.3f}"
            f"{result['minimum']:<12.3f}"
            f"{result['maximum']:<12.3f}"
            f"{overlap}/5{'':<10}"
            f"{average_unique_chunks:.1f}/5"
        )


# --------------------------------------------------
# 9. PRINT TOP-5 DOCUMENT DETAILS
# --------------------------------------------------

def print_document_details(all_results):

    print("\n\n" + "=" * 85)
    print("TOP-5 DOCUMENT COMPARISON")
    print("=" * 85)

    for result in all_results:

        print("\n" + "-" * 75)

        configuration = (
            "Deduplicated"
            if result.get("deduplicated")
            else f"Original {result['candidate_count']}"
        )

        print(
            f"Configuration: {configuration} | "
            f"Average latency: {result['average']:.3f}s"
        )

        print("-" * 75)

        for doc in result["top5_details"]:

            print(f"\nRank: {doc['rank']}")
            print(f"Passage ID: {doc['id']}")
            print(f"Source: {doc['source']}")
            print(f"Page: {doc['page']}")
            print(f"Chunk ID: {doc['chunk_id']}")
            print(f"Text preview: {doc['text_preview']}")

            print("." * 65)


# --------------------------------------------------
# 10. MAIN BENCHMARK EXECUTION
# --------------------------------------------------

def main():

    all_results = []

    for query_config in QUERIES:

        query_name = query_config["name"]

        if QUERY_FILTER and query_name != QUERY_FILTER:
            continue

        query = query_config["query"]
        target_keyword = query_config["target_keyword"]

        print("\n\n" + "#" * 75)
        print(f"QUERY: {query_name}")
        print(f"Question: {query}")
        print("#" * 75)

        base_passages, target_ids = retrieve_base_passages(
            query,
            target_keyword
        )

        if not base_passages:
            print("No documents retrieved. Skipping query.")
            continue

        # Inspect original retrieval results.
        inspect_retrieval_duplicates(base_passages)

        # Create deduplicated passages using existing backend utility.
        unique_passages = get_unique_reranked_chunks(base_passages)

        print("\n" + "=" * 65)
        print("DEDUPLICATION SUMMARY")
        print("=" * 65)

        print(f"Original passages:   {len(base_passages)}")
        print(f"Unique passages:     {len(unique_passages)}")
        print(
            f"Duplicates removed:  "
            f"{len(base_passages) - len(unique_passages)}"
        )

        query_results = []

        # Experiment A: original 20, 15 and 10 candidates.
        for candidate_count in CANDIDATE_COUNTS:

            if candidate_count > len(base_passages):

                print(
                    f"Skipping {candidate_count}: "
                    f"only {len(base_passages)} documents retrieved."
                )

                continue

            result = benchmark_candidate_count(
                query=query,
                candidate_count=candidate_count,
                base_passages=base_passages,
                target_ids=target_ids
            )

            result["query_name"] = query_name
            result["query"] = query
            result["deduplicated"] = False

            query_results.append(result)
            all_results.append(result)

        # Experiment B: all unique passages, without truncation.
        if unique_passages:

            print("\n\n" + "#" * 65)
            print("DEDUPLICATED CANDIDATE EXPERIMENT")
            print("#" * 65)

            result = benchmark_candidate_count(
                query=query,
                candidate_count=len(unique_passages),
                base_passages=unique_passages,
                target_ids=target_ids
            )

            result["query_name"] = query_name
            result["query"] = query
            result["deduplicated"] = True

            query_results.append(result)
            all_results.append(result)

        print_query_summary(query_name, query_results)

    return all_results


# --------------------------------------------------
# 11. EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    print("\nFlashRank Candidate Count and Deduplication Benchmark")
    print("-----------------------------------------------------")
    print(f"Model: {RERANKER_MODEL}")
    print(f"Original candidate counts: {CANDIDATE_COUNTS}")
    print(f"Final top K: {FINAL_TOP_K}")
    print(f"Query filter: {QUERY_FILTER}")

    results = main()

    if results:
        print_document_details(results)

    print("\nBenchmark completed.")
