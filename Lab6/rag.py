from openai import OpenAI
from dotenv import load_dotenv
import os
import sys
from retriver import load_documents
from embedding import create_embedding
from similarity import cosine_similarity


def main():
    sys.stdout.reconfigure(encoding="utf-8") # type: ignore

    # Load configuration from the .env file
    load_dotenv()

    documents = load_documents()

    document_embeddings = {}

    for filename, content in documents.items():
        document_embeddings[filename] = create_embedding(content)

    # Create a client that communicates with Groq
    client = OpenAI(
        base_url = os.getenv("BASE_URL"),
        api_key= os.getenv("GROQ_API_KEY")
    )

    print("=" * 40)
    print("      My AI Assistant")
    print("=" * 40)

    while True:
        user_input = input("\nYou : ")

        if user_input.lower() == "quit":
            print("\nAI  : Goodbye! Have a great day.")
            break

        question_embedding = create_embedding(user_input)

        best_score = -1

        best_document = None

        for filename, doc_embedding in document_embeddings.items():
            score = cosine_similarity(
                question_embedding,
                doc_embedding
            )

            if score > best_score:

                best_score = score

                best_document = filename

        if best_document is None:
            print("\nAI  : I don't have any documents to answer from.")
            continue

        context = documents[best_document]

        prompt = f"""
        Answer the question using only
        the following information.

        Context:

        {context}

        Question:

        {user_input}
        """

        # Send a questions to the AI model through Conversation Loop
        response = client.chat.completions.create(
            model = os.getenv("MODEL"), # type: ignore
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        # Display the response
        print(f"\nAI  : {response.choices[0].message.content}")


if __name__ == "__main__":
    main()
