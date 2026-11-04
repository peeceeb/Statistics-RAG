def generate_chat_response(query: str) -> str:

    """
    Generate a temporary response.

    Later, this function will invoke the RAG pipeline.
    """

    answer = f"""
    Thank you for your question!

    Your question:
    {query}

    This response was generated through FastAPI.

    The RAG pipeline is not connected yet.

    In the next phase, I will:
    1. Generate query embeddings.
    2. Retrieve relevant chunks from Qdrant.
    3. Rerank the retrieved chunks using FlashRank.
    4. Pass relevant context to the LLM.
    5. Generate an answer with document citations.
    """

    return answer