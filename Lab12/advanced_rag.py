from chunker import create_chunks
from embeddings import (
    embed_chunks,
    create_embedding
)

from chroma_store import (
    add_chunks,
    stored_texts,
    search as chroma_search
)

from retriever import keyword_score
from reranker import rerank


def build_rag(
    text,
    source
):

    chunks = create_chunks(
        text,
        source=source,
        chunk_size=200,
        overlap=50
    )

    # Only embed chunks that are new or whose text changed,
    # so restarting doesn't re-embed the whole document.
    stored = stored_texts([
        f'{chunk["source"]}_{chunk["chunk_id"]}'
        for chunk in chunks
    ])

    new_chunks = [
        chunk for chunk in chunks
        if stored.get(f'{chunk["source"]}_{chunk["chunk_id"]}') != chunk["text"]
    ]

    embedded_chunks = embed_chunks(
        new_chunks
    )

    if embedded_chunks:
        add_chunks(
            embedded_chunks
        )

    return embedded_chunks


def retrieve(
    query,
    top_k=5
):

    query_embedding = create_embedding(
        query
    )

    semantic_results = chroma_search(
        query_embedding,
        top_k=top_k
    )

    candidates = []

    documents = (
        semantic_results["documents"][0] # type: ignore
    )

    metadatas = (
        semantic_results["metadatas"][0] # type: ignore
    )

    distances = (
        semantic_results["distances"][0] # type: ignore
    )

    for i in range(len(documents)):

        text = documents[i]

        metadata = metadatas[i]

        distance = distances[i]

        semantic = 1 / (
            1 + distance
        )

        keyword = keyword_score(
            query,
            text
        )

        combined = (
            0.7 * semantic
            +
            0.3 * keyword
        )

        candidates.append({
            "text": text,
            "source": metadata["source"],
            "chunk_id": metadata["chunk_id"],
            "semantic_score": semantic,
            "keyword_score": keyword,
            "combined_score": combined
        })

    candidates.sort(
        key=lambda item:
        item["combined_score"],
        reverse=True
    )

    candidates = candidates[:top_k]
    print(                "\nRetrieved Sources Candidates:"
            )
    
    for candidate in candidates:
    
        print(
                    f"- "
                    f"{candidate['source']} "
                    f"(chunk "
                    f"{candidate['chunk_id']})"
                )
    
    return rerank(
        query,
        candidates,
        top_k=3
    )
