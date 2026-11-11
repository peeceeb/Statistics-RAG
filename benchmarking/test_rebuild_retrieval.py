
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
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

COLLECTION_NAME = "Internal_Docs_Rebuild"
QDRANT_URL = "http://localhost:6333"

DENSE_MODEL = "text-embedding-3-large"
SPARSE_MODEL = "Qdrant/bm25"

TOP_K = 5

QUERY = "What is data preparation for data mining?"


# --------------------------------------------------
# 2. Initialize services
# --------------------------------------------------

client = QdrantClient(
    url=QDRANT_URL,
    timeout=15
)

dense_embeddings = OpenAIEmbeddings(
    model=DENSE_MODEL,
    request_timeout=15,
    max_retries=0
)

sparse_embeddings = FastEmbedSparse(
    model_name=SPARSE_MODEL
)


# --------------------------------------------------
# 3. Inspect collection configuration
# --------------------------------------------------

collection = client.get_collection(
    COLLECTION_NAME
)

print("\n========== COLLECTION INFORMATION ==========")

print("Collection:", COLLECTION_NAME)
print("Status:", collection.status)
print("Points:", collection.points_count)

print(
    "Dense vectors:",
    collection.config.params.vectors
)

print(
    "Sparse vectors:",
    collection.config.params.sparse_vectors
)


# --------------------------------------------------
# 4. Dense retrieval
# --------------------------------------------------

dense_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=dense_embeddings,
    vector_name="",
    retrieval_mode=RetrievalMode.DENSE
)

print("\n========== DENSE RETRIEVAL ==========")

dense_results = dense_store.similarity_search(
    query=QUERY,
    k=TOP_K
)

for index, document in enumerate(dense_results, start=1):

    metadata = document.metadata

    print(f"\nRank: {index}")
    print("Source:", metadata.get("source"))
    print("Page:", metadata.get("page"))
    print("Chunk ID:", metadata.get("chunk_id"))
    print("Content:", document.page_content[:250])


# --------------------------------------------------
# 5. Hybrid retrieval
# --------------------------------------------------

hybrid_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=dense_embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name="",
    sparse_vector_name="sparse"
)

print("\n========== HYBRID RETRIEVAL ==========")

hybrid_results = hybrid_store.similarity_search(
    query=QUERY,
    k=TOP_K
)

for index, document in enumerate(hybrid_results, start=1):

    metadata = document.metadata

    print(f"\nRank: {index}")
    print("Source:", metadata.get("source"))
    print("Page:", metadata.get("page"))
    print("Chunk ID:", metadata.get("chunk_id"))
    print("Content:", document.page_content[:250])


# --------------------------------------------------
# 6. Final test summary
# --------------------------------------------------

print("\n========== TEST SUMMARY ==========")

print("Dense results:", len(dense_results))
print("Hybrid results:", len(hybrid_results))

print(
    "Dense unique chunks:",
    len({
        doc.metadata.get("chunk_id")
        for doc in dense_results
    })
)

print(
    "Hybrid unique chunks:",
    len({
        doc.metadata.get("chunk_id")
        for doc in hybrid_results
    })
)

if not dense_results:
    raise RuntimeError("Dense retrieval returned no results.")

if not hybrid_results:
    raise RuntimeError("Hybrid retrieval returned no results.")

print("\nRetrieval test completed successfully.")
