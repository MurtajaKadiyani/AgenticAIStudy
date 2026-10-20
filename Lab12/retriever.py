import numpy as np

from embeddings import create_embedding


def cosine_similarity(
    vector1,
    vector2
):

    v1 = np.array(vector1)
    v2 = np.array(vector2)

    return np.dot(v1, v2) / (
        np.linalg.norm(v1)
        *
        np.linalg.norm(v2)
    )


def semantic_search(
    query,
    chunks,
    top_k=5
):

    query_embedding = create_embedding(
        query
    )

    results = []

    for chunk in chunks:

        score = cosine_similarity(
            query_embedding,
            chunk["embedding"]
        )

        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "semantic_score": score
        })

    results.sort(
        key=lambda item:
        item["semantic_score"],
        reverse=True
    )

    return results[:top_k]


def keyword_score(
    query,
    text
):

    query_words = set(
        query.lower().split()
    )

    text_words = set(
        text.lower().split()
    )

    if not query_words:
        return 0.0

    matches = (
        query_words
        &
        text_words
    )

    return len(matches) / len(
        query_words
    )


def hybrid_search(
    query,
    chunks,
    top_k=5,
    semantic_weight=0.7,
    keyword_weight=0.3
):

    query_embedding = create_embedding(
        query
    )

    results = []

    for chunk in chunks:

        semantic = cosine_similarity(
            query_embedding,
            chunk["embedding"]
        )

        keyword = keyword_score(
            query,
            chunk["text"]
        )

        combined = (
            semantic * semantic_weight
            +
            keyword * keyword_weight
        )

        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "semantic_score": semantic,
            "keyword_score": keyword,
            "combined_score": combined
        })

    results.sort(
        key=lambda item:
        item["combined_score"],
        reverse=True
    )

    return results[:top_k]
