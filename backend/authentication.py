
import logging

from fastapi import Header, HTTPException
from google.auth.exceptions import TransportError
from google.auth.transport.requests import Request
from google.oauth2 import id_token

from backend.config.settings import GOOGLE_OAUTH_CLIENT_ID


logger = logging.getLogger(__name__)


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
        logger.warning("Authentication failed: ID token header missing.")
        raise HTTPException(
            status_code=401,
            detail="Authentication token is required."
        )

    if not GOOGLE_OAUTH_CLIENT_ID:
        logger.error("Google OAuth client ID is not configured.")
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

        logger.info("Google ID token verification succeeded.")

    except TransportError as exc:
        logger.error(
            "Google token verification service unavailable: %s",
            type(exc).__name__
        )
        raise HTTPException(
            status_code=503,
            detail="Unable to verify Google authentication at this time."
        )

    except ValueError as exc:
        error_text = str(exc).lower()

        if "expired" in error_text:
            reason = "expired"
        elif "audience" in error_text:
            reason = "audience_mismatch"
        elif "issuer" in error_text:
            reason = "issuer_mismatch"
        elif "signature" in error_text:
            reason = "signature_invalid"
        else:
            reason = "invalid_token"

        logger.warning(
            "Google ID token verification failed: reason=%s",
            reason
        )

        raise HTTPException(
            status_code=401,
            detail="Invalid or expired Google authentication token."
        )

    user_id = claims.get("sub")

    if not user_id:
        logger.warning("Verified Google ID token has no subject claim.")
        raise HTTPException(
            status_code=401,
            detail="Google authentication token has no user identity."
        )

    return {
        "user_id": user_id,
        "email": claims.get("email"),
        "name": claims.get("name")
    }
