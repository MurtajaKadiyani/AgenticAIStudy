def create_chunks(
        text,
        source,
        chunk_size=200,
        overlap=50
):

    words = text.split()

    chunks = []

    start = 0
    chunk_id = 0

    while start < len(words):

        end = start + chunk_size

        chunk_text = " ".join(words[start:end])

        chunks.append({
            "text": chunk_text,
            "source": source,
            "chunk_id": chunk_id
        })

        chunk_id += 1

        start += chunk_size - overlap

    return chunks
