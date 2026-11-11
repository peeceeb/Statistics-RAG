import sys
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from ingestion.extract_documents import ProcessDocuments

SOURCE_FOLDER = PROJECT_ROOT / "documents" / "internal"

processor = ProcessDocuments(
    folder_path=str(SOURCE_FOLDER),
    batch_size=2,
    chunk_size=1000,
    chunk_overlap=400,
    max_workers=2
)

pdf_files = sorted(SOURCE_FOLDER.glob("*.pdf"))

print("========== SOURCE INVENTORY ==========")
print("PDF count:", len(pdf_files))

all_chunk_ids = set()
duplicate_chunk_ids = set()
total_chunks = 0

for pdf_batch in processor.get_batches(iter(pdf_files)):

    documents = processor.process_batch(pdf_batch)

    counts = defaultdict(int)

    for document in documents:

        source = document.metadata.get("source")
        chunk_id = document.metadata.get("chunk_id")

        counts[source] += 1

        if chunk_id in all_chunk_ids:
            duplicate_chunk_ids.add(chunk_id)
        else:
            all_chunk_ids.add(chunk_id)

    for pdf_file in pdf_batch:

        count = counts[str(pdf_file)]
        total_chunks += count

        print(f"{pdf_file.name}: {count} chunks")

        if count == 0:
            print("WARNING: No chunks extracted!")

print("\n========== PREFLIGHT SUMMARY ==========")
print("PDFs discovered:", len(pdf_files))
print("Total chunks:", total_chunks)
print("Unique chunk IDs:", len(all_chunk_ids))
print("Duplicate chunk IDs:", len(duplicate_chunk_ids))

if duplicate_chunk_ids:
    print("WARNING: Duplicate chunk IDs detected!")

print("\nSource preflight completed.")