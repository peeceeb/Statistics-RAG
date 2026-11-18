
from fastapi import Header, HTTPException
from google.auth.exceptions import TransportError
from google.auth.transport.requests import Request
from google.oauth2 import id_token

from backend.config.settings import GOOGLE_OAUTH_CLIENT_ID


def get_current_user(
    x_end_user_id_token: str | None = Header(
        default=None,
        alias="X-End-User-ID-Token"
    )
) -> dict:
    """
    Verify the Google ID token and return the authenticated user identity.
    """

    if not x_end_user_id_token:
        raise HTTPException(
            status_code=401,
            detail="Authentication token is required."
        )

    if not GOOGLE_OAUTH_CLIENT_ID:
        raise HTTPException(
            status_code=500,
            detail="Google OAuth client ID is not configured."
        )

    try:
        claims = id_token.verify_oauth2_token(
            x_end_user_id_token,
            Request(),
            audience=GOOGLE_OAUTH_CLIENT_ID
        )

    except TransportError:
        raise HTTPException(
            status_code=503,
            detail="Unable to verify Google authentication at this time."
        )

    except ValueError:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired Google authentication token."
        )

    user_id = claims.get("sub")

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Google authentication token has no user identity."
        )

    return {
        "user_id": user_id,
        "email": claims.get("email"),
        "name": claims.get("name")
    }
