from memory_extractor import extract_memory

# memory = extract_memory(
#     "What is the current time?",
#     "The current time is 10 PM."
# )

memory = extract_memory(
    "I am leaning Agentic AI",
    "Thats Great! It has huge future."
)

print("Extracted Memory:")
print(memory)