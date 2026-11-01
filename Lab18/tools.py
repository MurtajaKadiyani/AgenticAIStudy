import os
import requests
import json
import time


from bs4 import BeautifulSoup
from ddgs import DDGS
from ddgs.exceptions import DDGSException
from dotenv import load_dotenv
from openai import OpenAI
from urllib.parse import urldefrag, urlparse, parse_qs

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("GROQ_API_KEY"),
    # Groq's per-minute token limit clears within seconds, so keep retrying
    max_retries=8
)

MODEL = os.getenv("MODEL")

research_notes = []
SEARCH_MIN_INTERVAL = 15
last_search_time = 0

source_texts = {}  # Text the agent has received, by page URL
MIN_EVIDENCE_LENGTH = 20  # Shorter quotes could match almost any page


def page_key(url):
    """Remove the parts of a URL that still point to the same page."""
    return urldefrag(str(url)).url.rstrip("/")


def remember_source_text(url, text):
    """Keep text the agent received so its evidence can be checked later."""
    key = page_key(url)
    source_texts[key] = f"{source_texts.get(key, '')}\n{text}"


def normalize_text(text):
    """Ignore differences in spacing, capitals, dashes and quote marks."""
    for variants, replacement in (("-‐‑‒–—", " "), ("‘’", "'"), ("“”", '"')):
        for variant in variants:
            text = text.replace(variant, replacement)

    return " ".join(str(text).split()).lower()

def search_web(query: str, max_results: int = 5):
    """
    Search the web using DuckDuckGo.

    Args:
        query: Search query.
        max_results: Maximum number of results.

    Returns:
        A list of search results.
    """

    global last_search_time

    max_results = min(int(max_results), 3)

    # Rapid searches get throttled, so leave a gap between them
    wait_time = SEARCH_MIN_INTERVAL - (time.monotonic() - last_search_time)

    if wait_time > 0:
        time.sleep(wait_time)

    last_search_time = time.monotonic()

    try:
        results = DDGS().text(
            query,
            max_results=max_results,
            backend="duckduckgo"
        )
    except DDGSException as error:
        raise RuntimeError(
            f"DuckDuckGo search failed: {error}. "
            "It may be throttling searches, so this does "
            "not mean no results exist. Try again later."
        ) from error

    for result in results:
        remember_source_text(result.get("href", ""), result.get("body", ""))

    return [
        {
            "title": result.get("title", ""),
            "url": result.get("href", ""),
            "snippet": result.get("body", "")
        }
        for result in results
    ]


def read_webpage(url: str):
    """
    Download a webpage and extract its text.

    Args:
        url: Webpage URL.

    Returns:
        Extracted webpage text or an error message.
    """

    if url.startswith("//"):
        url = "https:" + url

    # Extract the real URL from
    # DuckDuckGo redirect URL
    parsed_url = urlparse(url)

    if "duckduckgo.com" in parsed_url.netloc:
        params = parse_qs(parsed_url.query)

        if "uddg" in params:
            url = params["uddg"][0]

    try:

        response = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0"
            },
            timeout=20
        )

        response.raise_for_status()

    except requests.exceptions.RequestException as e:

        return {
            "status": "error",
            "url": url,
            "error": str(e)
        }

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    for element in soup(
        ["script", "style", "noscript"]
    ):
        element.decompose()

    text = soup.get_text(
        "\n",
        strip=True
    )

    # Long pages use up Groq's per-minute token limit quickly
    text = text[:3000]

    remember_source_text(url, text)

    return text


def add_research_note(
    source,
    finding,
    evidence
):
    """
    Store an important research finding.
IMPORTANT:
    This tool accepts exactly three arguments:

    source:
        The exact URL of the page you read.

    finding:
        The specific claim or conclusion
        extracted from the source.

    evidence:
        An exact quote copied word for word from
        that source, which supports the finding.
        Notes without an exact quote are rejected.

    Do not use title or summary.

    Returns:
        Confirmation message, or the reason for rejection.
    """

    quote = normalize_text(evidence).strip("\"' ")
    text = normalize_text(source_texts.get(page_key(source), ""))

    if len(quote) < MIN_EVIDENCE_LENGTH or quote not in text:
        return {
            "status": "rejected",
            "error": (
                "Note not stored. The evidence must be an exact quote "
                f"of at least {MIN_EVIDENCE_LENGTH} characters from text "
                "you received for this source URL. Copy it word for "
                "word, with no summary or added words, and try again."
            )
        }

    research_notes.append({
        "source": source,
        "finding": finding,
        "evidence": evidence,
        "verified": False
    })

    return {
        "status": "stored",
        "finding": finding
    }


def get_research_notes():
    """
    Return all research notes collected so far.
    """

    return [
        note.copy()
        for note in research_notes
    ]

def mark_note_verified(
    source,
    finding
):
    """
    Mark a matching research note as verified.
    """

    for note in research_notes:

        if (
            note["source"] == source
            and note["finding"] == finding
        ):
            note["verified"] = True

            return {
                "status": "verified",
                "finding": finding
            }

    return {
        "status": "not_found",
        "finding": finding
    }

def get_verified_research_notes():
    """
    Return only verified research notes.
    """

    return [
        note.copy()
        for note in research_notes
        if note.get("verified", False)
    ]

def verify_finding(
    source,
    finding,
    evidence
):
    """
    Check whether evidence supports a finding.
    """

    response = client.chat.completions.create(
        model=MODEL, # type: ignore
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an evidence verification assistant. "
                    "Determine whether the supplied evidence "
                    "supports the supplied finding. "
                    "Return JSON with exactly two fields: "
                    "supported and explanation. "
                    "supported must be true or false."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Source:\n{source}\n\n"
                    f"Finding:\n{finding}\n\n"
                    f"Evidence:\n{evidence}"
                )
            }
        ],
        response_format={"type": "json_object"}
    )

    result = json.loads(
        response.choices[0].message.content # type: ignore
    )

    if result["supported"]:

        mark_note_verified(
            source,
            finding
        )

    return result


