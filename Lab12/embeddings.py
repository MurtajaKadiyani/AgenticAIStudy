import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


client = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL"),
    api_key=os.getenv("OLLAMA_API_KEY")
)


MODEL = os.getenv("EMBED_MODEL")


def create_embedding(text):

    response = client.embeddings.create(
        model=MODEL, # type: ignore
        input=text
    )

    return response.data[0].embedding

def embed_chunks(chunks):

    if not chunks:
        return []

    # One request for all chunks instead of one per chunk.
    response = client.embeddings.create(
        model=MODEL, # type: ignore
        input=[chunk["text"] for chunk in chunks]
    )

    embeded_chunks = []

    for chunk, item in zip(chunks, response.data):

        embedding = item.embedding

        embeded_chunks.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "embedding": embedding
        })

    return embeded_chunks