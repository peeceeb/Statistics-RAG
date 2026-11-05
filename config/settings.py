import os

APP_NAME = "Statistical AI Assistant"

APP_DESCRIPTION = (
    "Your intelligent assistant for statistical analysis."
)

APP_ICON = "📚"

PAGE_LAYOUT = "wide"

BACKEND_API_URL = os.getenv(
    "BACKEND_API_URL",
    "http://localhost:8000"
)