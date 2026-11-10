
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from ingestion.extract_documents import ProcessDocuments

COLLECTION_NAME = "Internal_Docs"
QDRANT_URL = "http://localhost:6333"

PDF_PATH = (
    PROJECT_ROOT
    / "documents"
    / "internal"
    / "Enterprise_RAG_Sparse_Retrieval_Test_Document.pdf"
)

# Reprocess locally to reproduce the deterministic point IDs.
processor = ProcessDocuments(
    folder_path=str(PDF_PATH.parent),
    batch_size=1,
    chunk_size=1000,
    chunk_overlap=400,
    max_workers=1,
)

chunks = processor.process_batch([PDF_PATH])

ids = [
    str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.metadata["chunk_id"]))
    for chunk in chunks
]

client = QdrantClient(url=QDRANT_URL)

# Retrieve the exact points without modifying the collection.
records = client.retrieve(
    collection_name=COLLECTION_NAME,
    ids=ids,
    with_payload=True,
    with_vectors=True,
)

print("\n========== VECTOR VERIFICATION ==========")
print(f"Expected points: {len(ids)}")
print(f"Retrieved points: {len(records)}")

if len(records) != len(ids):
    raise RuntimeError("Some indexed points could not be retrieved.")

for record in records:
    vectors = record.vector

    if not isinstance(vectors, dict):
        raise RuntimeError(f"Unexpected vector format: {type(vectors)}")

    dense = vectors.get("")
    sparse = vectors.get("sparse")

    dense_ok = dense is not None and len(dense) == 3072

    sparse_ok = (
        sparse is not None
        and len(sparse.indices) > 0
        and len(sparse.values) > 0
    )

    print(f"\nPoint ID: {record.id}")
    print(f"Dense vector present: {dense_ok}")
    print(f"Dense dimensions: {len(dense) if dense is not None else 0}")
    print(f"Sparse vector present: {sparse_ok}")
    print(
        f"Sparse non-zero entries: "
        f"{len(sparse.indices) if sparse is not None else 0}"
    )

    if not dense_ok or not sparse_ok:
        raise RuntimeError(
            f"Hybrid vector verification failed for point {record.id}"
        )

print("\nSUCCESS: All points contain dense and sparse vectors.")
print("==========================================")
