import chromadb

client = chromadb.PersistentClient(
    path = "./chroma.db"
)

collection = (
    client.get_or_create_collection(
        name = "documents"
    )
)

def add_chunks(chunks):

    documents = []
    embeddings = []
    ids = []
    metadatas = []

    for chunk in chunks:

        documents.append(
            chunk["text"]
        )

        embeddings.append(
            chunk["embedding"]
        )

        ids.append(
            f'{chunk["source"]}_'
            f'{chunk["chunk_id"]}'
        )

        metadatas.append({
             "source": chunk["source"],
             "chunk_id": chunk["chunk_id"]
        })

    collection.upsert(
        ids = ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

def stored_texts(ids):

    # Map of id -> text for the chunks already in the store.
    existing = collection.get(
        ids = ids,
        include = ["documents"]
    )

    return dict(
        zip(
            existing["ids"],
            existing["documents"] # type: ignore
        )
    )

def search(
        query_embedding,
        top_k=5
):
    return collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=top_k
    )