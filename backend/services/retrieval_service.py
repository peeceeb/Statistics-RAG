
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

# --- STEP 1: Read the cabinet's spec sheet (covered earlier) ---
configured_dense_vectors = collection_info.config.params.vectors
configured_sparse_vectors = (collection_info.config.params.sparse_vectors or {})
# "or {}" = if there's no keyword index, use an empty box instead of "nothing"


# --- STEP 2: Find the label of the meaning-based drawer ---
if isinstance(configured_dense_vectors, dict):       # Multiple labeled drawers?

    if "dense" in configured_dense_vectors:          # Is one labeled "dense"?
        dense_vector_name = "dense"                  # Use it

    elif "" in configured_dense_vectors:             # Is there a blank-labeled one?
        dense_vector_name = ""                       # Use the blank label

    else:                                            # Nothing we recognize
        raise ValueError(                            # Stop with a clear error
            f"Collection {COLLECTION_NAME!r} "
            "has no supported dense vector."
        )

else:                                                # Only one unlabeled drawer
    dense_vector_name = ""                           # Use the blank label


# --- STEP 3: Start filling out the librarian's setup form ---
vector_store_options = {
    "client": qdrant_client,               # The connection to the database
    "collection_name": COLLECTION_NAME,    # Which cabinet to use
    "embedding": embedding_model,          # The "translator" that turns text into meaning-numbers
    "vector_name": dense_vector_name       # Which drawer holds the meaning-based data
}


# --- STEP 4: Does the cabinet ALSO have a keyword drawer? ---
if configured_sparse_vectors:              # Non-empty = yes, keyword index exists

    # Pick the keyword drawer's label:
    # use "sparse" if it exists, otherwise just grab the first one available
    sparse_vector_name = (
        "sparse"
        if "sparse" in configured_sparse_vectors
        else next(iter(configured_sparse_vectors))   # "next(iter(...))" = first item in the dict
    )

    # Load the keyword "translator" (turns text into keyword-weights, BM25-style)
    sparse_embedding_model = FastEmbedSparse(
        model_name=SPARSE_MODEL
    )

    # Add the keyword settings to the setup form
    vector_store_options.update(
        sparse_embedding=sparse_embedding_model,    # The keyword translator
        retrieval_mode=RetrievalMode.HYBRID,        # Use BOTH search styles
        sparse_vector_name=sparse_vector_name       # Which drawer holds keyword data
    )

    # A label for logs/reports describing how search works
    RETRIEVAL_METHOD = "Dense + BM25 + RRF"

else:
    # No keyword drawer, so fall back to meaning-only search
    vector_store_options["retrieval_mode"] = RetrievalMode.DENSE

    RETRIEVAL_METHOD = (
        "Dense (collection has no sparse vectors)"
    )

    # Let the developer know what happened
    print(
        f"Collection {COLLECTION_NAME!r} has no sparse vectors; "
        "using dense retrieval."
    )


# --- STEP 5: Hand the completed form to the librarian ---
vector_db = QdrantVectorStore(**vector_store_options)
# "**" unpacks the dictionary into named arguments, so this is equivalent to:
# QdrantVectorStore(client=..., collection_name=..., embedding=..., ...)

# @observe is a "decorator": it wraps the function below so that every time
# it runs, Langfuse automatically records a trace (start, end, duration, errors).
# name="..." is the label you'll see in the Langfuse dashboard.
@observe(name="internal-document-retrieval")
def search_documents(user_query: str):       # Takes the user's question as text

    # Open a more detailed "recording" for just the search step.
    # "with ... as" means: start recording here, and automatically stop
    # when the indented block ends (even if an error occurs).
    with langfuse.start_as_current_observation(
        as_type="retriever",                 # Tag this step as a retrieval/search step
        name="qdrant-document-search",       # Label shown in the dashboard
        input={                              # Log what went IN to the search
            "query": user_query,                 # The user's question
            "retrieval_top_k": RETRIEVAL_TOP_K   # How many results we asked for
        }
    ) as retrieval_observation:              # Handle to this recording, so we can add to it

        # THE ACTUAL SEARCH: the only line that does real work.
        # Converts the query to numbers, searches Qdrant (dense, or dense + keyword
        # if hybrid), and returns the top k best-matching documents.
        search_results = vector_db.similarity_search(
            query=user_query,
            k=RETRIEVAL_TOP_K                # e.g. 5 = "give me the 5 best matches"
        )

        # Add the OUTPUT to the recording: what came OUT of the search.
        retrieval_observation.update(
            output={
                "retrieved_count": len(search_results),   # How many docs came back
                "retrieval_method": RETRIEVAL_METHOD      # "Dense + BM25 + RRF" or "Dense..."
            }
        )

    # Recording is closed. Hand the documents back to whoever called this function.
    return search_results