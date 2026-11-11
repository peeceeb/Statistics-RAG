
import sys
import uuid
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    SparseVectorParams,
    Distance,
)
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

SOURCE_FOLDER = PROJECT_ROOT / "documents" / "internal"

EXISTING_COLLECTION = "Internal_Docs"
REBUILD_COLLECTION = "Internal_Docs_Rebuild"

QDRANT_URL = "http://localhost:6333"

DENSE_MODEL = "text-embedding-3-large"
SPARSE_MODEL = "Qdrant/bm25"

DENSE_VECTOR_SIZE = 3072
DENSE_VECTOR_NAME = ""
SPARSE_VECTOR_NAME = "sparse"

PDF_BATCH_SIZE = 2
EMBEDDING_BATCH_SIZE = 32
UPSERT_BATCH_SIZE = 32

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 400

TEST_MODE = False
TEST_MAX_PDFS = 1
TEST_MAX_CHUNKS = 3

SCROLL_PAGE_SIZE = 128

MANIFEST_PATH = (
    PROJECT_ROOT / "ingestion" / "rebuild_manifest.json"
)


# ==================================================
# 2. LOGGING
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ==================================================
# 3. INITIALIZE SERVICES
# ==================================================

from ingestion.extract_documents import ProcessDocuments

qdrant_client = QdrantClient(
    url=QDRANT_URL
)

dense_embeddings = OpenAIEmbeddings(
    model=DENSE_MODEL
)

sparse_embeddings = FastEmbedSparse(
    model_name=SPARSE_MODEL
)


# ==================================================
# 4. HELPER FUNCTIONS
# ==================================================

def generate_point_id(chunk_id: str) -> str:
    """
    Generate a deterministic UUID5 from a logical chunk ID.
    """

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            str(chunk_id)
        )
    )


def batched(items, batch_size):
    """
    Yield items in batches without loading all batches at once.
    """

    batch = []

    for item in items:
        batch.append(item)

        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def write_manifest(manifest: dict):
    """
    Save the manifest atomically.
    """

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_path = MANIFEST_PATH.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            indent=4,
            ensure_ascii=False
        )

    temporary_path.replace(MANIFEST_PATH)


# ==================================================
# 5. MANIFEST MANAGEMENT
# ==================================================

def initialize_manifest() -> tuple[dict, dict]:

    configuration = {
        "collection": REBUILD_COLLECTION,
        "dense_model": DENSE_MODEL,
        "sparse_model": SPARSE_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "dense_vector_size": DENSE_VECTOR_SIZE,
        "dense_vector_name": DENSE_VECTOR_NAME,
        "sparse_vector_name": SPARSE_VECTOR_NAME,
    }

    if MANIFEST_PATH.exists():

        with MANIFEST_PATH.open(
            "r",
            encoding="utf-8"
        ) as file:
            manifest = json.load(file)

        if manifest.get("configuration") != configuration:
            raise RuntimeError(
                "Manifest configuration differs from the current "
                "ingestion configuration. Inspect the manifest and "
                "rebuild collection before proceeding."
            )

    else:

        if qdrant_client.collection_exists(
            REBUILD_COLLECTION
        ):
            raise RuntimeError(
                "Rebuild collection already exists but no manifest "
                "was found. Inspect it before reusing it."
            )

        manifest = {
            "configuration": configuration,
            "runs": [],
        }

    run = {
        "started_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "mode": "TEST" if TEST_MODE else "FULL",
        "status": "running",
        "pdfs_completed": 0,
        "pdf_batches_completed": 0,
        "chunks_processed": 0,
        "points_upserted": 0,
        "processed_sources": [],
        "error": None,
    }

    manifest["runs"].append(run)

    write_manifest(manifest)

    return manifest, run


# ==================================================
# 6. COLLECTION MANAGEMENT
# ==================================================

def create_or_validate_collection():

    if not qdrant_client.collection_exists(
        REBUILD_COLLECTION
    ):

        logger.info(
            "Creating collection: %s",
            REBUILD_COLLECTION
        )

        qdrant_client.create_collection(
            collection_name=REBUILD_COLLECTION,

            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=DENSE_VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            },

            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams()
            }
        )

        logger.info("Rebuild collection created.")

    else:

        logger.info(
            "Reusing existing rebuild collection: %s",
            REBUILD_COLLECTION
        )

    collection = qdrant_client.get_collection(
        REBUILD_COLLECTION
    )

    vectors = collection.config.params.vectors
    sparse_vectors = collection.config.params.sparse_vectors

    if not isinstance(vectors, dict):
        raise RuntimeError(
            "Expected named dense vector configuration."
        )

    dense_config = vectors.get(DENSE_VECTOR_NAME)

    if dense_config is None:
        raise RuntimeError(
            "Expected dense vector configuration is missing."
        )

    if dense_config.size != DENSE_VECTOR_SIZE:
        raise RuntimeError(
            "Dense vector dimension does not match configuration."
        )

    if dense_config.distance != Distance.COSINE:
        raise RuntimeError(
            "Dense vector distance is not cosine."
        )

    if not sparse_vectors or SPARSE_VECTOR_NAME not in sparse_vectors:
        raise RuntimeError(
            "Expected sparse vector configuration is missing."
        )

    logger.info(
        "Collection validated | Status: %s | Existing points: %s",
        collection.status,
        collection.points_count
    )


# ==================================================
# 7. DENSE + SPARSE EMBEDDING
# ==================================================

def create_points(documents: list) -> list[PointStruct]:

    if not documents:
        return []

    texts = [
        document.page_content
        for document in documents
    ]

    chunk_ids = [
        document.metadata.get("chunk_id")
        for document in documents
    ]

    if any(not chunk_id for chunk_id in chunk_ids):
        raise ValueError(
            "At least one document is missing chunk_id."
        )

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(
            "Duplicate chunk IDs found inside an embedding batch."
        )

    # Generate dense embeddings.
    dense_vectors = dense_embeddings.embed_documents(
        texts
    )

    # Generate BM25 sparse vectors.
    sparse_vectors = sparse_embeddings.embed_documents(
        texts
    )

    if len(dense_vectors) != len(documents):
        raise RuntimeError(
            "Dense embedding count does not match document count."
        )

    if len(sparse_vectors) != len(documents):
        raise RuntimeError(
            "Sparse embedding count does not match document count."
        )

    points = []

    for document, dense_vector, sparse_vector in zip(
        documents,
        dense_vectors,
        sparse_vectors
    ):

        if len(dense_vector) != DENSE_VECTOR_SIZE:
            raise RuntimeError(
                "Unexpected dense embedding dimension."
            )

        chunk_id = document.metadata["chunk_id"]

        point = PointStruct(
            id=generate_point_id(chunk_id),

            vector={
                DENSE_VECTOR_NAME: dense_vector,

                SPARSE_VECTOR_NAME: {
                    "indices": sparse_vector.indices,
                    "values": sparse_vector.values,
                }
            },

            payload={
                "page_content": document.page_content,
                "metadata": dict(document.metadata),
            }
        )

        points.append(point)

    return points


# ==================================================
# 8. BATCH UPSERT
# ==================================================

def upsert_points(points: list[PointStruct]) -> int:

    total_upserted = 0

    for point_batch in batched(
        points,
        UPSERT_BATCH_SIZE
    ):

        result = qdrant_client.upsert(
            collection_name=REBUILD_COLLECTION,
            points=point_batch,
            wait=True
        )

        logger.info(
            "Upsert completed | Points: %s | Status: %s",
            len(point_batch),
            result.status
        )

        total_upserted += len(point_batch)

    return total_upserted


# ==================================================
# 9. FINAL COLLECTION VERIFICATION
# ==================================================

def verify_collection() -> dict:

    logger.info("Starting complete collection verification.")

    total_points = 0
    unique_chunk_ids = set()

    duplicate_points = 0
    missing_chunk_ids = 0
    invalid_payloads = 0
    invalid_dense_vectors = 0
    invalid_sparse_vectors = 0

    offset = None

    while True:

        records, next_offset = qdrant_client.scroll(
            collection_name=REBUILD_COLLECTION,
            limit=SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=True
        )

        for record in records:

            total_points += 1

            payload = record.payload or {}
            metadata = payload.get("metadata") or {}

            chunk_id = metadata.get("chunk_id")

            if not chunk_id:
                missing_chunk_ids += 1
            elif chunk_id in unique_chunk_ids:
                duplicate_points += 1
            else:
                unique_chunk_ids.add(chunk_id)

            if not isinstance(payload.get("page_content"), str):
                invalid_payloads += 1

            vectors = record.vector

            if not isinstance(vectors, dict):
                invalid_dense_vectors += 1
                invalid_sparse_vectors += 1
                continue

            dense_vector = vectors.get(DENSE_VECTOR_NAME)
            sparse_vector = vectors.get(SPARSE_VECTOR_NAME)

            if (
                dense_vector is None
                or len(dense_vector) != DENSE_VECTOR_SIZE
            ):
                invalid_dense_vectors += 1

            if isinstance(sparse_vector, dict):
                sparse_indices = sparse_vector.get("indices")
                sparse_values = sparse_vector.get("values")
            elif sparse_vector is not None:
                sparse_indices = sparse_vector.indices
                sparse_values = sparse_vector.values
            else:
                sparse_indices = None
                sparse_values = None

            if (
                not sparse_indices
                or not sparse_values
                or len(sparse_indices) != len(sparse_values)
            ):
                invalid_sparse_vectors += 1

        if next_offset is None:
            break

        offset = next_offset

    result = {
        "total_points": total_points,
        "unique_chunk_ids": len(unique_chunk_ids),
        "duplicate_points": duplicate_points,
        "missing_chunk_ids": missing_chunk_ids,
        "invalid_payloads": invalid_payloads,
        "invalid_dense_vectors": invalid_dense_vectors,
        "invalid_sparse_vectors": invalid_sparse_vectors,
    }

    logger.info("========== VERIFICATION SUMMARY ==========")

    for key, value in result.items():
        logger.info("%s: %s", key, value)

    if (
        duplicate_points
        or missing_chunk_ids
        or invalid_payloads
        or invalid_dense_vectors
        or invalid_sparse_vectors
    ):
        raise RuntimeError(
            "Collection verification failed. Inspect the summary."
        )

    if total_points != len(unique_chunk_ids):
        raise RuntimeError(
            "Point count does not match unique chunk ID count."
        )

    logger.info("Collection verification passed.")

    return result


# ==================================================
# 10. MAIN INGESTION WORKFLOW
# ==================================================

def main():

    # Confirm source collection exists.
    if not qdrant_client.collection_exists(
        EXISTING_COLLECTION
    ):
        raise RuntimeError(
            "Existing Internal_Docs collection was not found."
        )

    # Initialize manifest before creating the rebuild collection.
    manifest, run = initialize_manifest()

    try:

        create_or_validate_collection()

        # Initialize document processor.
        processor = ProcessDocuments(
            folder_path=str(SOURCE_FOLDER),
            batch_size=PDF_BATCH_SIZE,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            max_workers=2
        )

        pdf_files = sorted(
            SOURCE_FOLDER.glob("*.pdf")
        )

        if not pdf_files:
            raise FileNotFoundError(
                "No PDF documents found in internal folder."
            )

        logger.info(
            "PDF files discovered: %s",
            len(pdf_files)
        )

        # Test mode processes only the first PDF.
        if TEST_MODE:
            pdf_files = pdf_files[:TEST_MAX_PDFS]

        seen_chunk_ids = set()

        # Process PDFs in batches.
        for pdf_batch in processor.get_batches(
            iter(pdf_files)
        ):

            logger.info(
                "Processing PDF batch: %s",
                [pdf.name for pdf in pdf_batch]
            )

            documents = processor.process_batch(
                pdf_batch
            )

            if not documents:
                raise RuntimeError(
                    "Document processing returned no chunks."
                )

            # Ensure every PDF produced extractable chunks.
            chunks_by_source = defaultdict(int)

            for document in documents:
                chunks_by_source[
                    document.metadata.get("source")
                ] += 1

            for pdf_file in pdf_batch:

                if chunks_by_source[str(pdf_file)] == 0:
                    raise RuntimeError(
                        f"No chunks extracted from {pdf_file.name}. "
                        "Check whether the PDF needs OCR."
                    )

            # Restrict the first run to three chunks.
            if TEST_MODE:
                documents = documents[:TEST_MAX_CHUNKS]

            if not documents:
                raise RuntimeError(
                    "No chunks selected for ingestion."
                )

            # Detect duplicate logical chunks across this run.
            for document in documents:

                chunk_id = document.metadata["chunk_id"]

                if chunk_id in seen_chunk_ids:
                    raise RuntimeError(
                        f"Duplicate chunk ID detected: {chunk_id}"
                    )

                seen_chunk_ids.add(chunk_id)

            logger.info(
                "Chunks selected for ingestion: %s",
                len(documents)
            )

            # Generate embeddings in batches.
            for document_batch in batched(
                documents,
                EMBEDDING_BATCH_SIZE
            ):

                points = create_points(
                    document_batch
                )

                # Upsert in batches.
                upserted = upsert_points(
                    points
                )

                run["chunks_processed"] += len(document_batch)
                run["points_upserted"] += upserted

                write_manifest(manifest)

            run["pdf_batches_completed"] += 1
            run["pdfs_completed"] += len(pdf_batch)

            run["processed_sources"].extend(
                [pdf.name for pdf in pdf_batch]
            )

            write_manifest(manifest)

        logger.info(
            "Ingestion completed | Points upserted this run: %s",
            run["points_upserted"]
        )

        # Verify every point in the rebuild collection.
        verification = verify_collection()

        # In full mode, all points should correspond to this run.
        if not TEST_MODE:

            if (
                verification["total_points"]
                != run["points_upserted"]
            ):
                raise RuntimeError(
                    "Full ingestion point count does not match "
                    "the number of points upserted in this run."
                )

        run["verification"] = verification
        run["status"] = "completed"
        run["completed_at"] = datetime.now(
            timezone.utc
        ).isoformat()

        write_manifest(manifest)

        logger.info("Ingestion run completed successfully.")

        if TEST_MODE:
            logger.info(
                "TEST MODE is enabled. Full ingestion has NOT run."
            )

    except Exception as error:

        run["status"] = "failed"
        run["error"] = str(error)
        run["failed_at"] = datetime.now(
            timezone.utc
        ).isoformat()

        write_manifest(manifest)

        logger.exception(
            "Ingestion failed. Existing Internal_Docs remains untouched."
        )

        raise


# ==================================================
# 11. ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()
