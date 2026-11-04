import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(
    model=os.getenv("MODEL"), # type: ignore
    api_key=os.getenv("GROQ_API_KEY") # type: ignore
)
