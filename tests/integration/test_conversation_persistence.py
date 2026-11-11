
import uuid

from backend.database.connection import SessionLocal
from backend.repositories.conversation_repository import (
    get_or_create_conversation
)
from backend.repositories.message_repository import (
    save_message,
    get_conversation_messages
)


def test_conversation_persistence():

    db = SessionLocal()

    conversation_id = str(uuid.uuid4())
    user_id = "test_user_001"

    try:

        # 1. Create conversation
        conversation = get_or_create_conversation(
            db,
            user_id,
            conversation_id
        )

        print("\nConversation created:", conversation.id)

        # 2. Save user message
        save_message(
            db,
            conversation.id,
            "user",
            "Explain regression analysis."
        )

        # 3. Save assistant message
        save_message(
            db,
            conversation.id,
            "assistant",
            "Regression analysis studies relationships between variables.",
            sources=[
                {
                    "document_name": "Statistics.pdf",
                    "page_number": 10
                }
            ]
        )

        # 4. Retrieve conversation history
        messages = get_conversation_messages(
            db,
            conversation.id
        )

        print("\nRetrieved conversation history:")

        for message in messages:
            print(f"\nRole: {message.role}")
            print(f"Content: {message.content}")
            print(f"Sources: {message.sources}")

        # 5. Verify user isolation
        try:
            get_or_create_conversation(
                db,
                "another_user",
                conversation_id
            )
            raise AssertionError("Ownership check failed!")

        except PermissionError:
            print("\nUser ownership check passed.")

        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

        print("\nAll persistence tests passed!")

    finally:

        # Clean up test conversation
        existing = db.get(
            type(conversation),
            conversation.id
        )

        if existing:
            db.delete(existing)
            db.commit()

        db.close()


if __name__ == "__main__":
    test_conversation_persistence()
