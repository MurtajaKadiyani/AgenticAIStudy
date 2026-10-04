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

print("=" * 40)
print("      My AI Assistant")
print("=" * 40)

messages = []

while True:
    user_input = input("\nYou : ")

    # Save the user's message
    messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    if user_input.lower() == "quit":
        print("\nAI  : Goodbye! Have a great day.")
        break

    # Send a questions to the AI model through Conversation Loop
    response = client.chat.completions.create(
    model = os.getenv("MODEL"), # type: ignore
    messages = messages
    )

    ai_reply = response.choices[0].message.content
    print("\n AI :", ai_reply)

    # Save the AI's reply
    messages.append(
        {
            "role": "assistant",
            "content": ai_reply
        }
    )

    print("\n---------- Conversation History ----------")

    for message in messages:
        print(f"{message['role'].title()} : {message['content']}")

    print("------------------------------------------")