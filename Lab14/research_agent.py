from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("GROQ_API_KEY")
)

def research(task):

    prompt = f"""
    You are a Research Agent.

    Your job is to research the following task.

    Task:
    {task}

    Provide:

    1. Important concepts
    2. Key facts
    3. Relevant examples

    Keep the response concise and useful
    for another AI agent.
    """

    response = client.chat.completions.create(
            model=os.getenv("MODEL"), # type: ignore
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    
    return response.choices[0].message.content