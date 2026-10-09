import os

from dotenv import load_dotenv
from openai import OpenAI
from similarity import cosine_similarity

load_dotenv()

client = OpenAI(
     base_url=os.getenv("OLLAMA_BASE_URL"),
     api_key=os.getenv("OLLAMA_API_KEY")
)

EMBED_MODEL = os.getenv("EMBED_MODEL")

def create_embedding(content):
    embedding = client.embeddings.create(
    model=EMBED_MODEL, # type: ignore
    input=content
    ).data[0].embedding
    return embedding

