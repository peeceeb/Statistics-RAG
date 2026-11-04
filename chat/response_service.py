def generate_response(user_query: str) -> str:

    """
    Generate a temporary dummy response.

    Later this function will call the RAG backend.
    """

    response = f"""
    Thank you for your question!

    You asked:

    {user_query}

    This is a temporary AI response.

    Our RAG backend is not connected yet.

    Once integrated, I will retrieve relevant documents
    from Qdrant, rerank the retrieved chunks and generate
    an answer using the configured LLM.

    Source: Dummy response service.
    """

    return response