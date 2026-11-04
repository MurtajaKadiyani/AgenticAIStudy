from retriever import retriever

results = retriever.invoke(
    "What is MCP?"
)

for document in results:
    print("\n--- DOCUMENT ---")
    print(document.page_content)