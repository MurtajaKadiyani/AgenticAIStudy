import os
import re

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("GROQ_API_KEY")
)

def extract_memory(user_message, assistant_message):
    prompt = f"""
    You are a memory extraction system.
    Read the conversation below and determine whether
    it contains useful long-term information about the user.

    User:
   {user_message}

    Assistant:
   {assistant_message}

    Store information only if it will be useful
    in future conversations.

    Useful information includes:
   - preferences
   - interests
   - goals
   - skills
   - projects
   - important user facts

   Do NOT store:
   - temporary questions
   - current time
   - random calculations
   - one-time requests
   - tool results
   - passwords, codes, keys or any other secrets
   - casual conversation

   If useful information exists,
   return ONLY the fact that should be remembered.

   If there is nothing useful to remember,
   return:

   NONE
   """

    response = client.chat.completions.create(
            model=os.getenv("MODEL"), # type: ignore
            messages=[
                {
                    "role": "system",
                    "content": (
                         "You extract useful long-term memories."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
             ]
         )
     
    return response.choices[0].message.content.strip() # type: ignore


def normalize(text):
    return " ".join(text.lower().strip(" .").split())


def is_none(fact):
    return fact.strip().strip('."\'').upper() == "NONE"


def decide_fact(new_fact, existing_facts):

    # existing_facts: the closest stored facts, as dicts
    # with "id" and "fact".
    existing = "\n".join(
        f"#{item['id']}: {item['fact']}"
        for item in existing_facts
    )

    prompt = f"""
    You maintain a list of long-term facts about a user.

    Existing facts:
    {existing}

    New fact:
    {new_fact}

    Decide what to do with the new fact. Return exactly ONE line:

    DUPLICATE
        Every piece of information in the new fact is already
        stated by ONE existing fact. A fact that only mentions the
        same person or word is not a duplicate: "User's name is
        Murtaja" is NEW next to "Murtaja studies Python".

    REPLACE <id>[, <id> ...]
        The new fact updates, corrects or combines existing facts
        (e.g. a changed name, a finished or new topic, a fact that
        merges two old ones). List every existing fact it makes
        outdated. Only list a fact if the new fact still covers
        everything it said, or corrects it.

    NEW
        The new fact is about something the existing facts do not cover.

    Do not explain anything.
    """

    response = client.chat.completions.create(
            model=os.getenv("MODEL"), # type: ignore
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You keep a user's fact list accurate and free of duplicates."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

    decision = response.choices[0].message.content.strip() # type: ignore

    return parse_fact_decision(decision, existing_facts)


def parse_fact_decision(decision, existing_facts):

    text = decision.strip().splitlines()[0].upper() if decision.strip() else ""

    if text.startswith("DUPLICATE"):
        return "DUPLICATE", []

    if text.startswith("REPLACE"):

        valid_ids = {item["id"] for item in existing_facts}

        ids = [
            int(part)
            for part in text[len("REPLACE"):].replace("#", " ").replace(",", " ").split()
            if part.isdigit() and int(part) in valid_ids
        ]

        # Never touch a fact that wasn't offered.
        if ids:
            return "REPLACE", ids

    return "NEW", []


# Words that say nothing about what a fact is about.
STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was",
    "has", "have", "his", "her", "their", "they", "them", "who", "also",
    "now", "about", "part", "into", "user", "users",
    "learn", "learns", "learning", "learned",
    "study", "studies", "studying", "studied",
    "like", "likes", "liking",
}


def key_terms(text):
    words = re.findall(r"[a-z0-9]+", text.lower().replace("’", "'"))

    return {
        word for word in words
        if len(word) >= 3 and word not in STOPWORDS
    }


def safe_to_replace(old_fact, new_fact, all_facts):

    # A word in at least half of the user's facts (usually
    # their name) is not a key term: it would let any fact
    # "match" any other.
    counts = {}

    for item in all_facts:
        for term in key_terms(item["fact"]):
            counts[term] = counts.get(term, 0) + 1

    # count >= 2 too: with only two facts, "half" would
    # otherwise make every word common.
    common = {
        term for term, count in counts.items()
        if count >= 2 and count * 2 >= len(all_facts)
    }

    old_terms = key_terms(old_fact) - common

    # Nothing distinctive to check: trust the decision.
    if not old_terms:
        return True

    # The new fact must still mention what the old one was
    # about: "Python and MCP" keeps "python"; "learning MCP"
    # replacing "learning Python" does not.
    return bool(old_terms & key_terms(new_fact))


def store_fact(user_id, fact):

    # Imported here so this module's extract_memory
    # stays usable without a database.
    from memory import (
        create_embedding,
        rank_facts,
        add_fact,
        replace_facts,
        get_active_facts,
        RELEVANCE_THRESHOLD
    )

    embedding = create_embedding(fact)

    similar = rank_facts(
        user_id,
        embedding,
        top_k=3,
        threshold=RELEVANCE_THRESHOLD
    )

    # Nothing related stored yet.
    if not similar:
        add_fact(user_id, fact, embedding)
        return "NEW", []

    # Exact repeat: no need to ask the LLM. A similarity
    # cut-off is not safe here: "learning Python" vs
    # "learning Python and MCP" scores 0.954.
    if any(normalize(item["fact"]) == normalize(fact) for item in similar):
        return "DUPLICATE", []

    action, ids = decide_fact(fact, similar)

    if action == "REPLACE":

        # Second check on the LLM's decision: only retire an
        # old fact if the new one still covers its subject.
        # Otherwise keep both; a duplicate is better than a
        # lost fact.
        old_text = {item["id"]: item["fact"] for item in similar}
        all_facts = get_active_facts(user_id)

        ids = [
            fact_id for fact_id in ids
            if safe_to_replace(old_text[fact_id], fact, all_facts)
        ]

        if not ids:
            action = "NEW"

    if action == "REPLACE":

        replace_facts(user_id, ids, fact, embedding)

    elif action == "NEW":

        add_fact(user_id, fact, embedding)

    return action, ids