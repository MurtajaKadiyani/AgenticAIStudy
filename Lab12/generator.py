import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL = os.getenv("MODEL")

def generate_answer(
        query,
        results
):
    context = "\n\n".join(
        result["text"]
        for result in results
    )

    prompt = f"""
    Answer the user's question using
    only the provided context.
    
    Context:
    {context}
    
    Question:
    {query}
    
    Rules:
    
    1. Use only the provided context.
    2. Do not invent information.
    3. If the answer is not present,
       say that the information is
       not available.
    4. Give a clear and concise answer.
    """
    
    response = client.chat.completions.create(
        model=MODEL, # type: ignore
        reasoning_effort="low",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    return (
        response
        .choices[0]
        .message
        .content
        .strip() # type: ignore
    )
    
    