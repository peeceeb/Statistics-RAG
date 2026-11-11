import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import (
    QdrantVectorStore,
    FastEmbedSparse,
    RetrievalMode,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

COLLECTION_NAME = "Internal_Docs"
QDRANT_URL = "http://localhost:6333"
QUERY = "ERR-503"
TOP_K = 20

client = QdrantClient(url=QDRANT_URL)

dense_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

sparse_embeddings = FastEmbedSparse(
    model_name="Qdrant/bm25"
)

# Dense-only retrieval
dense_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=dense_embeddings,
    vector_name="",
    retrieval_mode=RetrievalMode.DENSE,
)

# Hybrid retrieval
hybrid_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=dense_embeddings,
    sparse_embedding=sparse_embeddings,
    vector_name="",
    sparse_vector_name="sparse",
    retrieval_mode=RetrievalMode.HYBRID,
)

dense_results = dense_store.similarity_search(
    query=QUERY,
    k=TOP_K,
)

hybrid_results = hybrid_store.similarity_search(
    query=QUERY,
    k=TOP_K,
)


def print_results(mode, results):
    print(f"\n{'=' * 70}")
    print(f"{mode} RESULTS")
    print(f"{'=' * 70}")

    for rank, doc in enumerate(results, start=1):
        source = Path(
            doc.metadata.get("source", "")
        ).name

        chunk_id = doc.metadata.get("chunk_id")

        exact_match = (
            QUERY.lower() in doc.page_content.lower()
        )

        print(
            f"Rank: {rank} | "
            f"Source: {source} | "
            f"Page: {doc.metadata.get('page')} | "
            f"Exact match: {exact_match}"
        )

        print(f"Chunk ID: {chunk_id}")


print(f"Query: {QUERY}")

print_results("DENSE-ONLY", dense_results)
print_results("HYBRID", hybrid_results)


def test_document_ranks(results):
    return [
        rank
        for rank, doc in enumerate(results, start=1)
        if Path(
            doc.metadata.get("source", "")
        ).name == "Enterprise_RAG_Sparse_Retrieval_Test_Document.pdf"
        and QUERY.lower() in doc.page_content.lower()
    ]


print("\n========== COMPARISON ==========")

print(
    "Dense-only exact-match ranks:",
    test_document_ranks(dense_results)
)

print(
    "Hybrid exact-match ranks:",
    test_document_ranks(hybrid_results)
)

print("================================")