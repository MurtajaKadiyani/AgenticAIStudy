from openai import OpenAI
from dotenv import load_dotenv
import os
import sys

sys.stdout.reconfigure(encoding="utf-8") # type: ignore

# Load configuration from the .env file
load_dotenv()

# Create a client that communicates with Ollama
client = OpenAI(
    base_url = os.getenv("BASE_URL"),
    api_key= os.getenv("GROQ_API_KEY")
)

# Send a question to the AI model
response = client.chat.completions.create(
    model = os.getenv("MODEL"), # type: ignore
    messages = [
        {
            "role": "user",
            "content": "What is Artificial Intelligence?"
        }
    ]
)

# Display the response
print(response.choices[0].message.content)