
import pytest
from fastapi import HTTPException
from google.auth.exceptions import TransportError

from backend import authentication


def test_missing_token():
    with pytest.raises(HTTPException) as exc:
        authentication.get_current_user(None)

    assert exc.value.status_code == 401


def test_invalid_token(monkeypatch):
    def mock_verify(*args, **kwargs):
        raise ValueError("Invalid token signature")

    monkeypatch.setattr(
        authentication.id_token,
        "verify_oauth2_token",
        mock_verify
    )

    with pytest.raises(HTTPException) as exc:
        authentication.get_current_user("invalid-token")

    assert exc.value.status_code == 401


def test_valid_token(monkeypatch):
    def mock_verify(*args, **kwargs):
        return {
            "sub": "test-user-123",
            "email": "test@example.com",
            "name": "Test User"
        }

    monkeypatch.setattr(
        authentication.id_token,
        "verify_oauth2_token",
        mock_verify
    )

    result = authentication.get_current_user("valid-token")

    assert result["user_id"] == "test-user-123"
    assert result["email"] == "test@example.com"
    assert result["name"] == "Test User"


def test_missing_subject_claim(monkeypatch):
    def mock_verify(*args, **kwargs):
        return {"email": "test@example.com"}

    monkeypatch.setattr(
        authentication.id_token,
        "verify_oauth2_token",
        mock_verify
    )

    with pytest.raises(HTTPException) as exc:
        authentication.get_current_user("valid-token")

    assert exc.value.status_code == 401


def test_google_verification_service_unavailable(monkeypatch):
    def mock_verify(*args, **kwargs):
        raise TransportError("Google verification unavailable")

    monkeypatch.setattr(
        authentication.id_token,
        "verify_oauth2_token",
        mock_verify
    )

    with pytest.raises(HTTPException) as exc:
        authentication.get_current_user("valid-token")

    assert exc.value.status_code == 503


def test_missing_oauth_client_id(monkeypatch):
    monkeypatch.setattr(
        authentication,
        "GOOGLE_OAUTH_CLIENT_ID",
        None
    )

    with pytest.raises(HTTPException) as exc:
        authentication.get_current_user("valid-token")

    assert exc.value.status_code == 500

