import json
import os
import sqlite3

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Tests set MEMORY_DB to their own file so they never
# write into the agent's real memory.
DATABASE = os.getenv("MEMORY_DB", "memory.db")

embedding_client = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL"),
    api_key=os.getenv("OLLAMA_API_KEY")
)

EMBED_MODEL = os.getenv("EMBED_MODEL")

# Facts scoring below this are not offered to the LLM
# as candidates to merge or replace when storing a new fact.
RELEVANCE_THRESHOLD = 0.50


# --------------------------------
# Database setup
# --------------------------------
#
# users     one row per person
# threads   one row per chat (like a ChatGPT conversation)
# messages  short-term memory: the chat history of a thread
# facts     long-term memory: shared across all of a
#           user's threads

def connect_db():
    connection = sqlite3.connect(DATABASE)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def create_database():
    connection = connect_db()
    cursor = connection.cursor()

    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS threads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL REFERENCES users(id),
        title TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id INTEGER NOT NULL REFERENCES threads(id),
        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- A replaced fact is kept, with superseded_by pointing
    -- at the fact that replaced it, so a bad replace can be
    -- seen and undone. Only rows with superseded_by NULL
    -- are active.
    CREATE TABLE IF NOT EXISTS facts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL REFERENCES users(id),
        fact TEXT NOT NULL,
        embedding TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        superseded_by INTEGER REFERENCES facts(id),
        superseded_at TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_messages_thread
        ON messages(thread_id, id);

    CREATE INDEX IF NOT EXISTS idx_facts_user
        ON facts(user_id);
    """)

    # Databases created before versioning lack these columns.
    columns = {
        row[1]
        for row in cursor.execute("PRAGMA table_info(facts)")
    }

    if "superseded_by" not in columns:
        cursor.execute(
            "ALTER TABLE facts ADD COLUMN superseded_by INTEGER REFERENCES facts(id)"
        )

    if "superseded_at" not in columns:
        cursor.execute(
            "ALTER TABLE facts ADD COLUMN superseded_at TIMESTAMP"
        )

    connection.commit()
    connection.close()


# --------------------------------
# Users and threads
# --------------------------------

def get_or_create_user(name):
    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute(
        "INSERT OR IGNORE INTO users (name) VALUES (?)",
        (name,)
    )

    cursor.execute(
        "SELECT id FROM users WHERE name = ?",
        (name,)
    )

    user_id = cursor.fetchone()[0]

    connection.commit()
    connection.close()

    return user_id


def create_thread(user_id, title=None):
    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute(
        "INSERT INTO threads (user_id, title) VALUES (?, ?)",
        (user_id, title)
    )

    thread_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return thread_id


def list_threads(user_id):
    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, title, created_at
        FROM threads
        WHERE user_id = ?
        ORDER BY id DESC
    """, (user_id,))

    rows = cursor.fetchall()

    connection.close()

    return [
        {"id": thread_id, "title": title, "created_at": created_at}
        for thread_id, title, created_at in rows
    ]


def set_thread_title(thread_id, title):
    connection = connect_db()

    connection.execute(
        "UPDATE threads SET title = ? WHERE id = ? AND title IS NULL",
        (title, thread_id)
    )

    connection.commit()
    connection.close()


# --------------------------------
# Messages (short-term, per thread)
# --------------------------------

def save_message(thread_id, role, content):
    connection = connect_db()

    connection.execute(
        "INSERT INTO messages (thread_id, role, content) VALUES (?, ?, ?)",
        (thread_id, role, content)
    )

    connection.commit()
    connection.close()


def get_recent_conversation(thread_id, limit=3):
    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT role, content
        FROM messages
        WHERE thread_id = ?
        ORDER BY id DESC
        LIMIT ?
    """, (thread_id, limit * 2))

    rows = cursor.fetchall()

    connection.close()

    # Oldest first, paired into exchanges.
    history = []

    for role, content in reversed(rows):

        if role == "user":
            history.append({"user": content, "assistant": ""})

        elif history:
            history[-1]["assistant"] = content

    return history[-limit:]


# --------------------------------
# Facts (long-term, per user)
# --------------------------------

def create_embedding(text):
    response = embedding_client.embeddings.create(
        model=EMBED_MODEL, # type: ignore
        input=text
    )

    return response.data[0].embedding


def cosine_similarity(vector1, vector2):
    v1 = np.array(vector1)
    v2 = np.array(vector2)

    return np.dot(v1, v2) / (
        np.linalg.norm(v1) *
        np.linalg.norm(v2)
    )


def rank_facts(user_id, embedding, top_k, threshold):
    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, fact, embedding, updated_at
        FROM facts
        WHERE user_id = ? AND superseded_by IS NULL
    """, (user_id,))

    rows = cursor.fetchall()

    connection.close()

    results = []

    for fact_id, fact, embedding_json, updated_at in rows:

        score = float(cosine_similarity(
            embedding,
            json.loads(embedding_json)
        ))

        if score >= threshold:
            results.append(
                {
                    "id": fact_id,
                    "fact": fact,
                    "updated_at": updated_at,
                    "score": score
                }
            )

    results.sort(
        key = lambda item: item["score"],
        reverse= True
    )

    return results[:top_k]


def search_facts(user_id, query, top_k=20):

    # No relevance cut-off here: a question and the fact that
    # answers it often score lower than unrelated pairs
    # ("What is my name?" vs "User's name is Murtaja" = 0.51,
    # "what is machine learning?" vs an MCP fact = 0.55).
    # Facts are short and few, so pass them all, most
    # relevant first, and only cap the count.
    return rank_facts(
        user_id,
        create_embedding(query),
        top_k,
        threshold=-1.0
    )


def add_fact(user_id, fact, embedding):
    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute(
        "INSERT INTO facts (user_id, fact, embedding) VALUES (?, ?, ?)",
        (user_id, fact, json.dumps(embedding))
    )

    fact_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return fact_id


def replace_facts(user_id, old_ids, fact, embedding):

    # Add the new fact and retire the old ones in one
    # transaction; nothing is overwritten or deleted.
    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute(
        "INSERT INTO facts (user_id, fact, embedding) VALUES (?, ?, ?)",
        (user_id, fact, json.dumps(embedding))
    )

    new_id = cursor.lastrowid

    cursor.executemany("""
        UPDATE facts
        SET superseded_by = ?, superseded_at = CURRENT_TIMESTAMP
        WHERE id = ? AND user_id = ? AND superseded_by IS NULL
    """, [(new_id, old_id, user_id) for old_id in old_ids])

    connection.commit()
    connection.close()

    return new_id


def get_active_facts(user_id):
    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, fact
        FROM facts
        WHERE user_id = ? AND superseded_by IS NULL
        ORDER BY id
    """, (user_id,))

    rows = cursor.fetchall()

    connection.close()

    return [{"id": fact_id, "fact": fact} for fact_id, fact in rows]


def get_fact_history(user_id):

    # Every fact, active or replaced, oldest first.
    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, fact, superseded_by, superseded_at
        FROM facts
        WHERE user_id = ?
        ORDER BY id
    """, (user_id,))

    rows = cursor.fetchall()

    connection.close()

    return [
        {
            "id": fact_id,
            "fact": fact,
            "superseded_by": superseded_by,
            "superseded_at": superseded_at
        }
        for fact_id, fact, superseded_by, superseded_at in rows
    ]


def undo_replace(user_id, new_id):

    # Bring back the facts that new_id replaced and remove
    # new_id. Returns how many facts were restored; 0 means
    # new_id did not replace anything.
    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute("""
        UPDATE facts
        SET superseded_by = NULL, superseded_at = NULL
        WHERE superseded_by = ? AND user_id = ?
    """, (new_id, user_id))

    restored = cursor.rowcount

    # Only remove an active fact; a fact that was itself
    # replaced later must be undone from the newest end.
    if restored:
        cursor.execute("""
            DELETE FROM facts
            WHERE id = ? AND user_id = ? AND superseded_by IS NULL
        """, (new_id, user_id))

        if cursor.rowcount == 0:
            connection.rollback()
            connection.close()
            return 0

    connection.commit()
    connection.close()

    return restored
