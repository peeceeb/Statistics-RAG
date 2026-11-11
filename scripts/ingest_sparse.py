
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import (
    FastEmbedSparse,
    QdrantVectorStore,
    RetrievalMode,
)

# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from ingestion.extract_documents import ProcessDocuments

COLLECTION_NAME = "Internal_Docs"
QDRANT_URL = "http://localhost:6333"

PDF_NAME = "Enterprise_RAG_Sparse_Retrieval_Test_Document.pdf"

PDF_PATH = PROJECT_ROOT / "documents" / "internal" / PDF_NAME


# --------------------------------------------------
# 2. Validate input PDF
# --------------------------------------------------

if not PDF_PATH.exists():
    raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

print(f"Selected PDF: {PDF_PATH.name}")


# --------------------------------------------------
# 3. Verify existing Qdrant collection
# --------------------------------------------------

client = QdrantClient(
    url=QDRANT_URL,
    timeout=30,
)

collection_before = client.get_collection(COLLECTION_NAME)

points_before = collection_before.points_count

dense_vectors = collection_before.config.params.vectors
sparse_vectors = collection_before.config.params.sparse_vectors

if "" not in dense_vectors:
    raise RuntimeError("Existing unnamed dense vector is missing.")

if "sparse" not in sparse_vectors:
    raise RuntimeError("Sparse vector schema is missing.")

print("\nExisting collection verified.")
print(f"Collection: {COLLECTION_NAME}")
print(f"Points before ingestion: {points_before}")
print(f"Dense vector configuration: {dense_vectors}")
print(f"Sparse vector configuration: {sparse_vectors}")


# --------------------------------------------------
# 4. Process only the selected PDF
# --------------------------------------------------

processor = ProcessDocuments(
    folder_path=str(PDF_PATH.parent),
    batch_size=1,
    chunk_size=1000,
    chunk_overlap=400,
    max_workers=1,
)

chunks = processor.process_batch([PDF_PATH])

if not chunks:
    raise RuntimeError("No chunks generated from the PDF.")

print(f"\nTotal chunks generated: {len(chunks)}")


# --------------------------------------------------
# 5. Generate deterministic Qdrant point IDs
# --------------------------------------------------

ids = []

for chunk in chunks:

    chunk_id = chunk.metadata.get("chunk_id")

    if not chunk_id:
        raise ValueError(
            "A chunk is missing its deterministic chunk_id."
        )

    point_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            str(chunk_id),
        )
    )

    ids.append(point_id)


if len(ids) != len(set(ids)):
    raise ValueError("Duplicate chunk IDs detected.")

print("Deterministic point IDs generated successfully.")


# --------------------------------------------------
# 6. Initialize dense and sparse embedding models
# --------------------------------------------------

dense_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

sparse_embeddings = FastEmbedSparse(
    model_name="Qdrant/bm25"
)

print("\nDense embedding model initialized.")
print("Sparse embedding model initialized.")


# --------------------------------------------------
# 7. Index into the existing Qdrant collection
# --------------------------------------------------

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=dense_embeddings,
    sparse_embedding=sparse_embeddings,
    ids=ids,
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name="",
    sparse_vector_name="sparse",
)

print("\nHybrid indexing completed.")


# --------------------------------------------------
# 8. Verify collection point count
# --------------------------------------------------

collection_after = client.get_collection(COLLECTION_NAME)

points_after = collection_after.points_count

print("\n========== INGESTION SUMMARY ==========")
print(f"Collection: {COLLECTION_NAME}")
print(f"Points before ingestion: {points_before}")
print(f"Points after ingestion: {points_after}")
print(f"Chunks processed: {len(chunks)}")
print("Dense vectors: Enabled")
print("Sparse vectors: Enabled")
print("=======================================")
