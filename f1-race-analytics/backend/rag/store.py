"""
Local persistent Chroma vector store for the knowledge base — deliberately
NOT Azure AI Search: this is a small, static, read-mostly document set (a
few dozen chunks), so a local embedded vector DB living entirely inside
this repo's own storage folder needs no separate service to provision, no
extra Azure resource to pay for, and no network dependency beyond the
embedding calls themselves.
"""

from pathlib import Path

import chromadb

from backend.rag.embeddings import embed_query, embed_texts

CHROMA_DIR = Path(__file__).parent.parent / "storage" / "chroma"
COLLECTION_NAME = "knowledge_base"


def _client() -> chromadb.ClientAPI:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _get_collection():
    # cosine similarity: OpenAI-style embeddings encode meaning in
    # direction, not magnitude — cosine is the standard distance for them
    # (Chroma's default is L2/Euclidean, which isn't).
    return _client().get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def build_index(chunks: list[dict]) -> None:
    """
    (Re)builds the collection from scratch. The knowledge base is small and
    static enough (a few markdown files, re-embedded in one batch call) that
    fully rebuilding on every run is simpler and cheap enough to prefer over
    diffing which chunks changed since the last build.
    """
    client = _client()
    if COLLECTION_NAME in {c.name for c in client.list_collections()}:
        client.delete_collection(COLLECTION_NAME)
    collection = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    if not chunks:
        return

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    ids = [f"{c['source']}::{c['section']}::{i}" for i, c in enumerate(chunks)]
    metadatas = [{"source": c["source"], "section": c["section"]} for c in chunks]

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)


def search_knowledge_base(query: str, k: int = 3) -> list[dict]:
    """Embeds `query` and returns the top-k most similar chunks, each with its source file/section and similarity distance (lower = closer)."""
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return []

    query_embedding = embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=min(k, count))

    return [
        {"text": doc, "source": meta["source"], "section": meta["section"], "distance": round(dist, 4)}
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ]
