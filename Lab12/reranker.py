import os
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("GROQ_API_KEY")
)


MODEL = os.getenv("MODEL")

def score_candidate(query, candidate):

    prompt = f"""
    You are a document relevance evaluator.

    User Query:
    {query}

    Document:
    {candidate["text"]}

    Give a relevance score from 0 to 100.

    100 = directly answers the query
    0 = completely irrelevant

    Return ONLY the number.
    """

    response = client.chat.completions.create(
                model=MODEL, # type: ignore
                reasoning_effort="low",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0
            )

    score_text = (
        response
        .choices[0]
        .message
        .content
        .strip() # type: ignore
    )

    try:
        score = float(score_text)
    except ValueError:
        score = 0

    return {
        **candidate,
        "rerank_score": score
    }

def rerank(
        query,
        candidates,
        top_k = 3
):

    if not candidates:
        return []

    # Score all candidates at the same time instead of
    # waiting for each LLM call in turn.
    with ThreadPoolExecutor(max_workers=len(candidates)) as executor:
        scored = list(
            executor.map(
                lambda candidate: score_candidate(query, candidate),
                candidates
            )
        )

    scored.sort(
         key=lambda item:
         item["rerank_score"],
         reverse= True
    )

    return scored[:top_k]
