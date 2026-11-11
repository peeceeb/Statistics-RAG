
import os

from dotenv import load_dotenv


# Load environment variables from the project .env file.
load_dotenv()


# Centralized OpenAI model configuration.
CHAT_MODEL = os.getenv(
    "OPENAI_CHAT_MODEL",
    "gpt-5.5"
)
