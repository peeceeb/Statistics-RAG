
import csv
import time
import statistics
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_qdrant import (
    QdrantVectorStore,
    FastEmbedSparse,
    RetrievalMode,
)
from langchain_openai import OpenAIEmbeddings


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

QDRANT_URL = "http://localhost:6333"

COLLECTIONS = [
    "Internal_Docs",
    "Internal_Docs_Rebuild",
]

DENSE_MODEL = "text-embedding-3-large"
SPARSE_MODEL = "Qdrant/bm25"

TOP_K = 5
REPETITIONS = 3

OUTPUT_FILE = (
    PROJECT_ROOT
    / "benchmarking"
    / "internal_collection_comparison.csv"
)

QUERIES = {
    "exact_keyword": "ERR-503",
    "conceptual": "Explain regression analysis and its applications",
    "technical": "How does hybrid retrieval combine dense and sparse search?",
    "broad_content": "What are the important concepts covered in these documents?",
    "comparison": "What are the differences between dense and hybrid retrieval?",
}


# --------------------------------------------------
# 2. Initialize services
# --------------------------------------------------

client = QdrantClient(
    url=QDRANT_URL,
    timeout=120,
)

dense_embeddings = OpenAIEmbeddings(
    model=DENSE_MODEL,
    request_timeout=60,
    max_retries=2,
)

sparse_embeddings = FastEmbedSparse(
    model_name=SPARSE_MODEL,
)


# --------------------------------------------------
# 3. Initialize vector stores
# --------------------------------------------------

stores = {}

for collection_name in COLLECTIONS:

    collection = client.get_collection(collection_name)

    print(f"\nCollection: {collection_name}")
    print("Status:", collection.status)
    print("Points:", collection.points_count)

    stores[collection_name] = {
        "dense": QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=dense_embeddings,
            vector_name="",
            retrieval_mode=RetrievalMode.DENSE,
        ),
        "hybrid": QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="",
            sparse_vector_name="sparse",
        ),
    }


# --------------------------------------------------
# 4. Retrieval helper
# --------------------------------------------------

def run_retrieval(store, query):

    start_time = time.perf_counter()

    documents = store.similarity_search(
        query=query,
        k=TOP_K,
    )

    elapsed = time.perf_counter() - start_time

    chunk_ids = [
        doc.metadata.get("chunk_id")
        for doc in documents
    ]

    valid_ids = [
        chunk_id
        for chunk_id in chunk_ids
        if chunk_id
    ]

    unique_ids = set(valid_ids)

    result = {
        "latency_seconds": elapsed,
        "documents": documents,
        "chunk_ids": chunk_ids,
        "unique_chunk_count": len(unique_ids),
        "duplicate_count": len(valid_ids) - len(unique_ids),
        "missing_chunk_id_count": len(chunk_ids) - len(valid_ids),
    }

    return result


# --------------------------------------------------
# 5. Warm-up
# --------------------------------------------------

print("\n========== WARM-UP ==========")

warmup_query = "What is data preparation for data mining?"

for collection_name in COLLECTIONS:
    for mode in ["dense", "hybrid"]:

        run_retrieval(
            stores[collection_name][mode],
            warmup_query,
        )

        print(f"Warm-up completed: {collection_name} | {mode}")


# --------------------------------------------------
# 6. Benchmark execution
# --------------------------------------------------

records = []
detailed_results = {}

print("\n========== BENCHMARK ==========")

for query_name, query in QUERIES.items():

    print(f"\nQuery: {query_name}")
    print("Text:", query)

    for repetition in range(1, REPETITIONS + 1):

        # Alternate collection order to reduce ordering bias.
        collection_order = (
            COLLECTIONS
            if repetition % 2 == 1
            else list(reversed(COLLECTIONS))
        )

        for collection_name in collection_order:

            for mode in ["dense", "hybrid"]:

                result = run_retrieval(
                    stores[collection_name][mode],
                    query,
                )

                chunk_ids = result["chunk_ids"]

                record = {
                    "query_name": query_name,
                    "query": query,
                    "repetition": repetition,
                    "collection": collection_name,
                    "mode": mode,
                    "latency_seconds": round(
                        result["latency_seconds"], 4
                    ),
                    "result_count": len(result["documents"]),
                    "unique_chunk_count": result["unique_chunk_count"],
                    "duplicate_count": result["duplicate_count"],
                    "missing_chunk_id_count": result[
                        "missing_chunk_id_count"
                    ],
                    "chunk_ids": "|".join(
                        str(x) if x else ""
                        for x in chunk_ids
                    ),
                }

                records.append(record)

                key = (
                    query_name,
                    repetition,
                    collection_name,
                    mode,
                )

                detailed_results[key] = result

                print(
                    f"{collection_name:22} | "
                    f"{mode:6} | "
                    f"Run {repetition} | "
                    f"{result['latency_seconds']:.3f}s | "
                    f"Unique: {result['unique_chunk_count']} | "
                    f"Duplicates: {result['duplicate_count']}"
                )


# --------------------------------------------------
# 7. Summary statistics
# --------------------------------------------------

print("\n========== SUMMARY ==========")

summary = {}

for collection_name in COLLECTIONS:
    for mode in ["dense", "hybrid"]:

        selected = [
            record
            for record in records
            if record["collection"] == collection_name
            and record["mode"] == mode
        ]

        latencies = [
            record["latency_seconds"]
            for record in selected
        ]

        unique_counts = [
            record["unique_chunk_count"]
            for record in selected
        ]

        duplicate_counts = [
            record["duplicate_count"]
            for record in selected
        ]

        summary[(collection_name, mode)] = {
            "avg_latency": statistics.mean(latencies),
            "median_latency": statistics.median(latencies),
            "avg_unique_chunks": statistics.mean(unique_counts),
            "avg_duplicates": statistics.mean(duplicate_counts),
        }

        print(f"\n{collection_name} | {mode}")

        print(
            "Average latency:",
            f"{statistics.mean(latencies):.3f}s",
        )

        print(
            "Median latency:",
            f"{statistics.median(latencies):.3f}s",
        )

        print(
            "Average unique chunks:",
            f"{statistics.mean(unique_counts):.2f}",
        )

        print(
            "Average duplicates:",
            f"{statistics.mean(duplicate_counts):.2f}",
        )


# --------------------------------------------------
# 8. Compare chunk overlap
# --------------------------------------------------

print("\n========== TOP-5 CHUNK OVERLAP ==========")

for query_name in QUERIES:

    for mode in ["dense", "hybrid"]:

        original = detailed_results[
            (query_name, 1, "Internal_Docs", mode)
        ]

        rebuilt = detailed_results[
            (query_name, 1, "Internal_Docs_Rebuild", mode)
        ]

        original_ids = {
            x for x in original["chunk_ids"] if x
        }

        rebuilt_ids = {
            x for x in rebuilt["chunk_ids"] if x
        }

        overlap = original_ids.intersection(rebuilt_ids)

        print(
            f"{query_name:18} | {mode:6} | "
            f"Shared chunks: {len(overlap)} | "
            f"Original unique: {len(original_ids)} | "
            f"Rebuilt unique: {len(rebuilt_ids)}"
        )


# --------------------------------------------------
# 9. Export detailed results
# --------------------------------------------------

fieldnames = [
    "query_name",
    "query",
    "repetition",
    "collection",
    "mode",
    "latency_seconds",
    "result_count",
    "unique_chunk_count",
    "duplicate_count",
    "missing_chunk_id_count",
    "chunk_ids",
]

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(records)


print("\n========== COMPLETED ==========")
print("Total benchmark records:", len(records))
print("CSV saved to:", OUTPUT_FILE)
