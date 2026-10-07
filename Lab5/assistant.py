from openai import OpenAI
from dotenv import load_dotenv
import os
import sys
from tools import read_text_file
from tool_manager import execute_tool

sys.stdout.reconfigure(encoding="utf-8") # type: ignore

# Load configuration from the .env file
load_dotenv()

# Create a client that communicates with Ollama
client = OpenAI(
    base_url = os.getenv("BASE_URL"),
    api_key= os.getenv("GROQ_API_KEY")
)

def ask_ai(messages):
    response = client.chat.completions.create(
        model = os.getenv("MODEL"), # type: ignore
        messages = messages # type: ignore
    )
    return response.choices[0].message.content


def summarize_or_explain(instruction, filename):
    file_content = read_text_file("data/" + filename)

    prompt = f"""
    {instruction} the following document.

    Document:

    {file_content}
    """

    return ask_ai([
        {
            "role": "system",
            "content": "You are a helpful AI assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
    ])


def answer_question(filename, question):
    file_content = read_text_file("data/" + filename)

    prompt = f"""
    You are given document.

    Answer the user's question using
    only the information present
    in the document.

    If the answer is not available,
    say:
    'I couldn't find that information
    in the document.'

    Document:

    {file_content}

    Question:

    {question}
    """

    return ask_ai([
        {
            "role": "system",
            "content": "You are a helpful AI assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
    ])


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

    if user_input.lower() == "quit":
        print("\nAI  : Goodbye! Have a great day.")
        break

    text = user_input.lower()

    if text.startswith("summarize"):
        filename = user_input[10:].strip()
        ai_reply = summarize_or_explain("Summarize", filename)
        print("\n AI :", ai_reply)
        continue

    elif text.startswith("explain"):
        filename = user_input[8:].strip()
        ai_reply = summarize_or_explain("Explain", filename)
        print("\n AI :", ai_reply)
        continue
    
    elif text.startswith("ask"):
        parts = user_input.split(maxsplit=2)
        # print(parts)
        filename = parts[1]
        question = parts[2]
        ai_reply = answer_question(filename, question)
        print("\n AI :", ai_reply)
        continue

    tool_result = execute_tool(user_input)

    if tool_result:
        print("\nAI :", tool_result)
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": str(tool_result)})
        continue

    # Save the user's message
    messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Send a questions to the AI model through Conversation Loop
    ai_reply = ask_ai(messages)
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