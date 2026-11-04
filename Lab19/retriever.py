import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

loader = TextLoader(
    "notes.txt",
    encoding="utf-8"
)

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

embeddings = OllamaEmbeddings(
    model=os.getenv("EMBED_MODEL"), # type: ignore
    # OLLAMA_BASE_URL ends in /v1 (OpenAI-style); OllamaEmbeddings needs the root URL
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").removesuffix("/v1")
)

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)