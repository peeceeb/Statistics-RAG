
from backend.services.retrieval_service import (
    search_documents,
    qdrant_client,
    langfuse,
    RETRIEVAL_METHOD
)

from backend.services.reranker_service import rerank_documents


# --------------------------------------------------
# STEP 1: DISPLAY RESULTS
# --------------------------------------------------

def display_results(results: list):

    print("\nFINAL RERANKED RESULTS")
    print("=" * 70)

    if not results:
        print("No relevant documents found.")
        return

    for rank, result in enumerate(results, start=1):

        print(f"\nResult {rank}")
        print("-" * 70)

        print(f"Rerank Score : {result['score']:.6f}")
        print(f"Source       : {result['meta'].get('source')}")
        print(f"Page         : {result['meta'].get('page')}")
        print(f"Chunk ID     : {result['meta'].get('chunk_id')}")

        print("\nContent:")
        print(result["text"][:500])


# --------------------------------------------------
# STEP 2: MAIN APPLICATION
# --------------------------------------------------

def main():

    print("\nRAG APPLICATION STARTED")
    print(f"{RETRIEVAL_METHOD} + FlashRank Reranking")
    print("Type 'exit' to stop.")

    try:

        while True:

            user_query = input("\nAsk something: ").strip()

            if user_query.lower() in ["exit", "quit"]:
                break

            if not user_query:
                print("Please enter a question.")
                continue

            try:

                # Step 1: Retrieve top 20 documents from Qdrant
                retrieved_documents = search_documents(user_query)

                print(
                    f"\n{RETRIEVAL_METHOD} retrieved "
                    f"{len(retrieved_documents)} chunks."
                )

                # Step 2: Rerank documents using FlashRank
                final_results = rerank_documents(
                    user_query,
                    retrieved_documents
                )

                # Step 3: Display final top 5 results
                display_results(final_results)

            except Exception as exc:

                print(f"Retrieval error: {exc}")

    finally:

        langfuse.flush()
        qdrant_client.close()


if __name__ == "__main__":
    main()
