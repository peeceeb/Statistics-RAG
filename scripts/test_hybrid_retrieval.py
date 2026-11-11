
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.retrieval_service import search_documents


def main():
    query = "ERR-503"

    print(f"\nQuery: {query}")
    print("Executing hybrid retrieval...\n")

    results = search_documents(query)

    print(f"Documents retrieved: {len(results)}")

    for index, doc in enumerate(results, start=1):
        print(f"\n{'=' * 60}")
        print(f"Result: {index}")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Page: {doc.metadata.get('page')}")
        print(f"Chunk ID: {doc.metadata.get('chunk_id')}")
        print(f"Content:\n{doc.page_content[:700]}")

        if query.lower() in doc.page_content.lower():
            print("Exact keyword match: YES")
        else:
            print("Exact keyword match: NO")

    print("\nHybrid retrieval test completed.")


if __name__ == "__main__":
    main()
