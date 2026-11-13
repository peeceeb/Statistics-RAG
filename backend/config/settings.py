
import os

from dotenv import load_dotenv


# --------------------------------------------------
# Environment Configuration
# --------------------------------------------------

# Load environment variables from the project .env file.
load_dotenv()


# --------------------------------------------------
# OpenAI Configuration
# --------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CHAT_MODEL = os.getenv(
    "OPENAI_CHAT_MODEL",
    "gpt-5.5"
)


# --------------------------------------------------
# PostgreSQL Configuration
# --------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL")


# --------------------------------------------------
# Qdrant Configuration
# --------------------------------------------------

# Local development defaults to the dedicated Enterprise_RAG instance.
# Docker deployment can override this through environment variables.

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6334"
)
