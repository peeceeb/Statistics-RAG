from pathlib import Path
from pprint import pprint
import sys

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ingestion.extract_documents import ProcessDocuments
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv


# Find project root
load_dotenv(project_root / ".env")


# Initialize document processor
retrieval = ProcessDocuments(
    folder_path=str(project_root / "documents" / "internal"),
    batch_size=2,
    chunk_size=1000,
    chunk_overlap=400,
    max_workers=2
)


print("---------------- 1. PROCESSING DOCUMENTS ----------------")

# Step 1: Discover PDF files
pdf_files = retrieval.get_pdf_files()

print("PDF files discovered:")

for pdf in pdf_files:
    print(pdf)


print("---------------- 2. GENERATING BATCHES ----------------")

# Step 2: Generate batches
batches = retrieval.get_batches(
    retrieval.get_pdf_files()
)


# Step 3: Process all batches
all_chunks = []

for i, batch in enumerate(batches, start=1):

    print(f"\n---------------- BATCH {i} ----------------")

    pprint(batch)

    # Load, split and sanitize documents
    chunks = retrieval.process_batch(batch)

    # Collect processed chunks
    all_chunks.extend(chunks)


print("---------------- 3. COUNTING EMBEDDING TOKENS ----------------")

# Count tokens before sending chunks to the embedding model
token_stats = retrieval.count_embedding_tokens(
    documents=all_chunks,
    model="text-embedding-3-large"
)

print(f"Total chunks: {token_stats['total_chunks']}")
print(f"Total embedding tokens: {token_stats['total_tokens']}")
print(
    f"Average tokens per chunk: "
    f"{token_stats['average_tokens_per_chunk']}"
)
print(
    f"Maximum tokens in a chunk: "
    f"{token_stats['max_tokens_per_chunk']}"
)


print("---------------- 4. LOADING DOCUMENTS INTO VECTOR DB ----------------")

# Step 4: Load documents into Vector DB
print("\n---------- Load Documents into Vector DB -------------")


# Step 5: Initialize the embedding model
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)


# Step 6: Create a Qdrant vector store from processed chunks
try:

    vector_store = QdrantVectorStore.from_documents(
        documents=all_chunks,
        embedding=embedding_model,
        url="http://localhost:6333",
        collection_name="Internal_Docs"
    )

    print(
        "Indexing of Internal_Docs Completed...."
    )

except Exception as exc:

    print(f"Qdrant connection failed: {exc}")

    print(
        "PDFs were loaded and chunked successfully, "
        "but vector indexing was skipped."
    )