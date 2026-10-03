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
    api_key= os.getenv("API_KEY")
)

print("=" * 40)
print("      My AI Assistant")
print("=" * 40)

while True:
    user_input = input("\nYou : ")

    if user_input.lower() == "quit":
        print("\nAI  : Goodbye! Have a great day.")
        break

    # Send a questions to the AI model through Conversation Loop
    response = client.chat.completions.create(
    model = os.getenv("MODEL"), # type: ignore
    messages = [
        {
            "role": "user",
            "content": user_input
        }
    ]
)

   # Display the response
    print(f"\nAI  : {response.choices[0].message.content}")