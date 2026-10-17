import os

# Use a separate database so test data never
# ends up in the agent's real memory.
os.environ["MEMORY_DB"] = "test_memory.db"

if os.path.exists("test_memory.db"):
    os.remove("test_memory.db")

from memory import (
    create_database,
    get_or_create_user,
    create_thread,
    save_message,
    get_recent_conversation,
    search_facts
)
from memory_extractor import store_fact

create_database()

user_id = get_or_create_user("test_user")

# Long-term facts, shared across threads.
print("\nStoring facts")
print("-------------")

for fact in [
    "The user's name is Murtaja.",
    "Murtaja is learning Python.",
    "Murtaja is learning Python.",            # exact duplicate
    "Murtaja's name is Murtaja.",             # same meaning
    "Murtaja is learning Python and MCP.",    # should replace/merge
    "Murtaja is studying Newton's Law.",
    "Murtaja finished Newton's Law and now studies optics.",  # should replace
]:
    print(store_fact(user_id, fact), "<-", fact)

import sqlite3

print("\nAll stored facts")
print("----------------")

for row in sqlite3.connect("test_memory.db").execute("SELECT id, fact FROM facts"):
    print(row)

results = search_facts(user_id, "What is my name?")

print("\nRelevant Facts")
print("--------------")

for result in results:
    print(round(result["score"], 2), result["fact"])

# Short-term history stays inside its own thread.
thread_a = create_thread(user_id, "dice chat")
thread_b = create_thread(user_id, "study chat")

save_message(thread_a, "user", "roll a dice")
save_message(thread_a, "assistant", "You rolled a 4.")
save_message(thread_b, "user", "What am I studying?")
save_message(thread_b, "assistant", "Python and MCP.")

print("\nThread A history:", get_recent_conversation(thread_a))
print("Thread B history:", get_recent_conversation(thread_b))
