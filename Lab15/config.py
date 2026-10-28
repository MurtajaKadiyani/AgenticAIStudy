import os

from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL","openai/gpt-oss-20b")

BASE_URL = os.getenv("BASE_URL","https://api.groq.com/openai/v1")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Add it to your .env file or environment."
    )

MAX_REVISIONS = int(
    os.getenv(
        "MAX_REVISIONS",
        "8"
    )
)


MAX_RETRIES = int(
    os.getenv(
        "MAX_RETRIES",
        "3"
    )
)


TIMEOUT = int(
    os.getenv(
        "TIMEOUT",
        "60"
    )
)
