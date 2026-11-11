
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.retrieval_service import search_documents
from backend.services.reranker_service import rerank_documents


QUERY = "ERR-503"
EXPECTED_DOCUMENT = "Enterprise_RAG_Sparse_Retrieval_Test_Document.pdf"
EXPECTED_CHUNK_ID = (
    "77d34e6b8799772d02fc40eff35fd9a7f8b614aaaca3057bf9b1aca1c85f4c1c"
)


def main():

    # Step 1: Hybrid retrieval
    print(f"\nQuery: {QUERY}")
    print("\nExecuting hybrid retrieval...")

    retrieved_documents = search_documents(QUERY)

    print(f"Hybrid retrieved: {len(retrieved_documents)}")

    if not retrieved_documents:
        print("No documents retrieved.")
        return

    # Step 2: FlashRank reranking
    print("\nExecuting FlashRank reranking...")

    reranked_documents = rerank_documents(
        QUERY,
        retrieved_documents
    )

    print(f"Reranked results: {len(reranked_documents)}")

    # Step 3: Inspect reranked results
    print("\n========== FINAL RERANKED RESULTS ==========")

    exact_match_found = False
    expected_chunk_found = False

    for rank, doc in enumerate(reranked_documents, start=1):

        # Reranker returns text and metadata under these keys.
        content = doc.get("text", "")
        metadata = doc.get("meta") or {}

        source = metadata.get("source", "")
        source_name = Path(source).name if source else "Unknown"

        chunk_id = metadata.get("chunk_id")

        exact_match = QUERY.lower() in content.lower()

        if exact_match:
            exact_match_found = True

        if chunk_id == EXPECTED_CHUNK_ID:
            expected_chunk_found = True

        print(f"\nRank: {rank}")
        print(f"Source: {source_name}")
        print(f"Page: {metadata.get('page')}")
        print(f"Chunk ID: {chunk_id}")
        print(f"Reranker score: {doc.get('score')}")
        print(f"Exact ERR-503 match: {exact_match}")

        if chunk_id == EXPECTED_CHUNK_ID:
            print("Expected test chunk: YES")
            print("\nChunk content:")
            print(content[:700])

    # Step 4: Final verification
    print("\n========== VERIFICATION SUMMARY ==========")

    print(f"Hybrid results: {len(retrieved_documents)}")
    print(f"Final reranked results: {len(reranked_documents)}")
    print(f"Expected chunk found: {expected_chunk_found}")
    print(f"Exact ERR-503 match found: {exact_match_found}")

    if expected_chunk_found and exact_match_found:
        print("\nSUCCESS: Hybrid retrieval and reranking verified.")
    else:
        print("\nCHECK REQUIRED: Expected chunk or exact match missing.")

    print("===========================================")


if __name__ == "__main__":
    main()
