from uuid import uuid4

from backend.database.connection import SessionLocal
from backend.services.chat_service import generate_chat_response


def test_generate_chat_response():
    db = SessionLocal()

    try:
        user_id = str(uuid4())
        conversation_id = str(uuid4())

        response = generate_chat_response(
            db=db,
            user_id=user_id,
            conversation_id=conversation_id,
            user_query="Explain regression analysis."
        )

        print("\nANSWER:")
        print(response["answer"])

        print("\nSOURCES:")
        for source in response["sources"]:
            print(source)

        assert response["answer"]
        assert isinstance(response["sources"], list)

    finally:
        db.close()