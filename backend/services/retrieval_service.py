
from dotenv import load_dotenv

from langchain_qdrant import (
    QdrantVectorStore,
    FastEmbedSparse,
    RetrievalMode
)
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langfuse import get_client, observe

load_dotenv()

# Configuration
COLLECTION_NAME = "Internal_Docs"
QDRANT_URL = "http://localhost:6333"

RETRIEVAL_TOP_K = 20

EMBEDDING_MODEL = "text-embedding-3-large"
SPARSE_MODEL = "Qdrant/bm25"

# Initialize Langfuse
langfuse = get_client()

# Initialize embeddings
embedding_model = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    request_timeout=15,
    max_retries=0
)

# Connect to Qdrant
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    timeout=10
)

# Inspect collection configuration
collection_info = qdrant_client.get_collection(COLLECTION_NAME)

configured_dense_vectors = collection_info.config.params.vectors
configured_sparse_vectors = (
    collection_info.config.params.sparse_vectors or {}
)

# Identify dense vector name
if isinstance(configured_dense_vectors, dict):

    if "dense" in configured_dense_vectors:
        dense_vector_name = "dense"

    elif "" in configured_dense_vectors:
        dense_vector_name = ""

    else:
        raise ValueError(
            f"Collection {COLLECTION_NAME!r} "
            "has no supported dense vector."
        )

else:
    dense_vector_name = ""

# Configure vector store
vector_store_options = {
    "client": qdrant_client,
    "collection_name": COLLECTION_NAME,
    "embedding": embedding_model,
    "vector_name": dense_vector_name
}

# Enable hybrid retrieval if sparse vectors exist
if configured_sparse_vectors:

    sparse_vector_name = (
        "sparse"
        if "sparse" in configured_sparse_vectors
        else next(iter(configured_sparse_vectors))
    )

    sparse_embedding_model = FastEmbedSparse(
        model_name=SPARSE_MODEL
    )

    vector_store_options.update(
        sparse_embedding=sparse_embedding_model,
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_vector_name=sparse_vector_name
    )

    RETRIEVAL_METHOD = "Dense + BM25 + RRF"

else:

    vector_store_options["retrieval_mode"] = RetrievalMode.DENSE

    RETRIEVAL_METHOD = (
        "Dense (collection has no sparse vectors)"
    )

    print(
        f"Collection {COLLECTION_NAME!r} has no sparse vectors; "
        "using dense retrieval."
    )

# Initialize Qdrant vector store
vector_db = QdrantVectorStore(**vector_store_options)


# Retrieve relevant documents
@observe(name="internal-document-retrieval")
def search_documents(user_query: str):

    with langfuse.start_as_current_observation(
        as_type="retriever",
        name="qdrant-document-search",
        input={
            "query": user_query,
            "retrieval_top_k": RETRIEVAL_TOP_K
        }
    ) as retrieval_observation:

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
