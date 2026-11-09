import os

APP_NAME = "AI Based Multi DocumentAssistant"

APP_DESCRIPTION = (
    "Your intelligent assistant for managing and interacting with multiple documents seamlessly."
)

APP_ICON = "📚"

PAGE_LAYOUT = "wide"

BACKEND_API_URL = os.getenv(
    "BACKEND_API_URL",
    "http://localhost:8000"
)