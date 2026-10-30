import os
from crewai import LLM
from dotenv import load_dotenv

load_dotenv()

llm = LLM(
    model=os.getenv("MODEL"), # type: ignore
    provider="openai",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=6000
)

# Groq rate limits are per model, so the manager and memory each use their own
manager_llm = LLM(
    model=os.getenv("MANAGER_MODEL"), # type: ignore
    provider="openai",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=6000
)

memory_llm = LLM(
    model=os.getenv("MEMORY_MODEL"), # type: ignore
    provider="openai",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=800 # this model allows 1000 output tokens per minute
)

