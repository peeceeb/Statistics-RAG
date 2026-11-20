
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routes import chat


app = FastAPI()
app.include_router(chat.router)

client = TestClient(app)


def mock_authenticated_user():
    return {
        "user_id": "user-A",
        "email": "test@example.com",
        "name": "Test User"
    }


def mock_database():
    return None


app.dependency_overrides[chat.get_current_user] = mock_authenticated_user
app.dependency_overrides[chat.get_db] = mock_database


def test_query_rejects_another_user(monkeypatch):
    def unexpected_chat_call(*args, **kwargs):
        raise AssertionError("Chat service must not be called")

    monkeypatch.setattr(
        chat,
        "generate_chat_response",
        unexpected_chat_call
    )

    response = client.post(
        "/api/v1/chat/query",
        json={
            "conversation_id": "test-conversation",
            "user_id": "user-B",
            "query": "Test query"
        }
    )

    assert response.status_code == 403


def test_conversations_reject_another_user(monkeypatch):
    def unexpected_repository_call(*args, **kwargs):
        raise AssertionError("Conversation repository must not be called")

    monkeypatch.setattr(
        chat,
        "get_user_conversations",
        unexpected_repository_call
    )

    response = client.get(
        "/api/v1/chat/conversations",
        params={"user_id": "user-B"}
    )

    assert response.status_code == 403


def test_messages_reject_another_user(monkeypatch):
    def unexpected_repository_call(*args, **kwargs):
        raise AssertionError("Conversation repository must not be called")

    monkeypatch.setattr(
        chat,
        "get_user_conversation",
        unexpected_repository_call
    )

    response = client.get(
        "/api/v1/chat/conversations/test-conversation/messages",
        params={"user_id": "user-B"}
    )

    assert response.status_code == 403
