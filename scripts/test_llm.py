
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.llm_service import generate_answer


def main():
    answer = generate_answer(
        user_query="What is regression analysis?",
        retrieved_chunks=[
            {
                "text": (
                    "Regression analysis examines the relationship "
                    "between a dependent variable and one or more "
                    "independent variables."
                ),
                "meta": {
                    "source": "Statistics.pdf",
                    "page": 10
                }
            }
        ]
    )

    print(answer)


if __name__ == "__main__":
    main()
