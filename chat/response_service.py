
import requests

from config.settings import BACKEND_API_URL


def generate_response(
    user_query: str,
    conversation_id: str,
    user_id: str
) -> dict:
    """
    Sends the user's query to FastAPI
    and returns the generated answer and sources.
    """

    url = f"{BACKEND_API_URL}/api/v1/chat/query"

    payload = {
        "conversation_id": conversation_id,
        "user_id": user_id,
        "query": user_query
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            "The backend took too long to respond."
        ) from exc

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Could not connect to the FastAPI backend. "
            "Check whether the backend server is running."
        ) from exc

    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"Backend returned an HTTP error: {exc}"
        ) from exc

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"An error occurred while communicating with the backend: {exc}"
        ) from exc

    except ValueError as exc:
        raise RuntimeError(
            "The backend returned an invalid JSON response."
        ) from exc


def get_conversations(user_id: str) -> list:
    """
    Retrieves all saved conversations belonging to a user.
    """

    url = f"{BACKEND_API_URL}/api/v1/chat/conversations"

    try:
        response = requests.get(
            url,
            params={"user_id": user_id},
            timeout=30
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Could not retrieve conversations: {exc}"
        ) from exc

    except ValueError as exc:
        raise RuntimeError(
            "The backend returned an invalid JSON response."
        ) from exc


def get_conversation_messages(
    user_id: str,
    conversation_id: str
) -> list:
    """
    Retrieves messages belonging to a specific conversation.
    """

    url = (
        f"{BACKEND_API_URL}/api/v1/chat/conversations/"
        f"{conversation_id}/messages"
    )

    try:
        response = requests.get(
            url,
            params={"user_id": user_id},
            timeout=30
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Could not retrieve conversation messages: {exc}"
        ) from exc

    except ValueError as exc:
        raise RuntimeError(
            "The backend returned an invalid JSON response."
        ) from exc
