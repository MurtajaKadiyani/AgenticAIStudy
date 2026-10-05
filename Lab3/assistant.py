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

roles = {
    "1": "You are a friendly school teacher. Explain every concept using simple language and real-life examples.",

    "2": "You are a senior Python developer. Explain programming concepts clearly and always include Python examples.",

    "3": "You are an experienced travel guide. Recommend places, food, transportation and travel tips.",

    "4": "You are a motivational coach. Encourage the user and give practical advice with a positive attitude.",

    "5": "You are a professional interviewer. Ask one interview question at a time and provide feedback after each answer."
}

print("\nChoose Your Assistant\n")

print("1. Teacher")
print("2. Python Expert")
print("3. Travel Guide")
print("4. Motivational Coach")
print("5. Interviewer")

while True:
    choice = input("\nEnter your choice : ")

    if choice in roles:
        break

    print("Incorrect choice, please type choices between 1-5.")

messages = [
    {
        "role": "system",
        "content": roles[choice]
    }
]

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
    messages = messages # type: ignore
    )

    ai_reply = response.choices[0].message.content
    print("\n AI :", ai_reply)

    # Save the AI's reply
    messages.append(
        {
            "role": "assistant",
            "content": ai_reply # type: ignore
        }
    )

    # print("\n---------- Conversation History ----------")

    # for message in messages:
    #     print(f"{message['role'].title()} : {message['content']}")

    # print("------------------------------------------")