from backend.services.llm_service import generate_answer

answer = generate_answer(
    user_query="What is regression analysis?",
    retrieved_chunks=[
        {
            "text": "Regression analysis examines the relationship "
                    "between a dependent variable and one or more "
                    "independent variables.",
            "meta": {
                "source": "Statistics.pdf",
                "page": 10
            }
        }
    ]
)

print(answer)