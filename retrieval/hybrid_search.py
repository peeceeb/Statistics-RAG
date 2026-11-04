from dotenv import load_dotenv

from langchain_qdrant import (
    QdrantVectorStore,
    FastEmbedSparse,
    RetrievalMode
)
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

from flashrank import Ranker, RerankRequest
from langfuse import get_client, observe

load_dotenv()

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

COLLECTION_NAME = "Internal_Docs"
QDRANT_URL = "http://localhost:6333"

RETRIEVAL_TOP_K = 20
FINAL_TOP_K = 5

EMBEDDING_MODEL = "text-embedding-3-large"
SPARSE_MODEL = "Qdrant/bm25"
RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"


# --------------------------------------------------
# INITIALIZE MODELS
# --------------------------------------------------

langfuse = get_client()

# Dense embeddings
embedding_model = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    request_timeout=15,
    max_retries=0
)

# Qdrant connection
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    timeout=10
)

collection_info = qdrant_client.get_collection(COLLECTION_NAME)
configured_dense_vectors = collection_info.config.params.vectors
configured_sparse_vectors = collection_info.config.params.sparse_vectors or {}

if isinstance(configured_dense_vectors, dict):
    if "dense" in configured_dense_vectors:
        dense_vector_name = "dense"
    elif "" in configured_dense_vectors:
        dense_vector_name = ""
    else:
        raise ValueError(
            f"Collection {COLLECTION_NAME!r} has no supported dense vector."
        )
else:
    dense_vector_name = ""

vector_store_options = {
    "client": qdrant_client,
    "collection_name": COLLECTION_NAME,
    "embedding": embedding_model,
    "vector_name": dense_vector_name,
}

if configured_sparse_vectors:
    sparse_vector_name = (
        "sparse" if "sparse" in configured_sparse_vectors
        else next(iter(configured_sparse_vectors))
    )
    sparse_embedding_model = FastEmbedSparse(
        model_name=SPARSE_MODEL
    )
    vector_store_options.update(
        sparse_embedding=sparse_embedding_model,
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_vector_name=sparse_vector_name,
    )
    RETRIEVAL_METHOD = "Dense + BM25 + RRF"
else:
    vector_store_options["retrieval_mode"] = RetrievalMode.DENSE
    RETRIEVAL_METHOD = "Dense (collection has no sparse vectors)"
    print(
        f"Collection {COLLECTION_NAME!r} has no sparse vectors; "
        "using dense retrieval."
    )

vector_db = QdrantVectorStore(**vector_store_options)

# FlashRank reranker
ranker = Ranker(
    model_name=RERANKER_MODEL
)


# --------------------------------------------------
# STEP 1: COLLECTION-AWARE SEARCH
# --------------------------------------------------

@observe(name="internal-document-retrieval")
def search_documents(user_query):

    with langfuse.start_as_current_observation(
        as_type="retriever",
        name="qdrant-document-search",
        input={
            "query": user_query,
            "retrieval_top_k": RETRIEVAL_TOP_K
        }
    ) as retrieval_observation:

        # Use hybrid retrieval when the collection has sparse vectors;
        # otherwise use its available dense vectors.
        search_results = vector_db.similarity_search(
            query=user_query,
            k=RETRIEVAL_TOP_K
        )

        retrieval_observation.update(
            output={
                "retrieved_count": len(search_results),
                "retrieval_method": RETRIEVAL_METHOD
            }
        )

    return search_results


# --------------------------------------------------
# STEP 2: RERANK TOP 20 CHUNKS
# --------------------------------------------------

@observe(name="flashrank-reranking")
def rerank_documents(user_query, search_results):

    with langfuse.start_as_current_observation(
        as_type="span",
        name="flashrank-reranking",
        input={
            "query": user_query,
            "candidate_count": len(search_results),
            "reranker_model": RERANKER_MODEL
        }
    ) as rerank_observation:

        # Convert LangChain Documents into FlashRank passages
        passages = [
            {
                "id": index,
                "text": doc.page_content,
                "meta": doc.metadata
            }
            for index, doc in enumerate(search_results)
        ]

        # Create reranking request
        rerank_request = RerankRequest(
            query=user_query,
            passages=passages
        )

        # Calculate relevance scores and reorder documents
        reranked_results = ranker.rerank(rerank_request)

        # Keep only the final top 5
        final_results = reranked_results[:FINAL_TOP_K]

        rerank_observation.update(
            output={
                "reranked_count": len(reranked_results),
                "final_count": len(final_results),
                "results": [
                    {
                        "rerank_score": float(result["score"]),
                        "source": result["meta"].get("source"),
                        "page": result["meta"].get("page")
                    }
                    for result in final_results
                ]
            }
        )

    return final_results


# --------------------------------------------------
# STEP 3: DISPLAY RESULTS
# --------------------------------------------------

def display_results(results):

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
# MAIN APPLICATION
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

                # Step 1: Retrieve top 20 with the supported collection mode
                retrieved_documents = search_documents(user_query)

                print(
                    f"\n{RETRIEVAL_METHOD} retrieved "
                    f"{len(retrieved_documents)} chunks."
                )

                # Step 2: Rerank retrieved chunks
                final_results = rerank_documents(
                    user_query,
                    retrieved_documents
                )

                # Step 3: Display final top 5
                display_results(final_results)

            except Exception as exc:

                print(f"Retrieval error: {exc}")

            finally:

                langfuse.flush()

    finally:

        langfuse.flush()
        qdrant_client.close()


if __name__ == "__main__":
    main()