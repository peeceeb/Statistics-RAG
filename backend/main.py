
import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import text
from qdrant_client import QdrantClient

from backend.routes.chat import router as chat_router
from backend.database.connection import engine
from backend.config.settings import QDRANT_URL


# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Health Check Configuration
# --------------------------------------------------

HEALTH_CHECK_COLLECTION = "Internal_Docs"

qdrant_health_client = QdrantClient(
    url=QDRANT_URL,
    timeout=3
)


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="RAG API",
    description="Backend API for RAG AI Assistant",
    version="1.0.0"
)

app.include_router(chat_router)


# --------------------------------------------------
# Root Endpoint
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "RAG API is running"
    }


# --------------------------------------------------
# Health Check Endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():

    dependencies = {}

    # PostgreSQL health check
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        dependencies["postgresql"] = "connected"

    except Exception:
        logger.exception("PostgreSQL health check failed")
        dependencies["postgresql"] = "unavailable"

    # Qdrant health check
    try:
        collection_info = qdrant_health_client.get_collection(
            HEALTH_CHECK_COLLECTION
        )

        dependencies["qdrant"] = "connected"

        dependencies["internal_docs"] = {
            "status": getattr(
                collection_info.status,
                "value",
                str(collection_info.status)
            ),
            "points_count": collection_info.points_count
        }

    except Exception:
        logger.exception("Qdrant health check failed")
        dependencies["qdrant"] = "unavailable"
        dependencies["internal_docs"] = {
            "status": "unavailable"
        }

    is_healthy = (
        dependencies["postgresql"] == "connected"
        and dependencies["qdrant"] == "connected"
    )

    response = {
        "status": "healthy" if is_healthy else "unhealthy",
        "dependencies": dependencies
    }

    if not is_healthy:
        logger.warning("Application health check reported unhealthy")

        return JSONResponse(
            status_code=503,
            content=response
        )

    return response
