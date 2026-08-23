"""
Builds/rebuilds the knowledge base's Chroma index from
backend/knowledge_base/*.md. Run this once after adding/editing any
knowledge_base file, and again any time you change EMBEDDING_DEPLOYMENT.

Run with:
    python -m backend.rag.build_index
"""

from backend.rag.chunking import chunk_knowledge_base
from backend.rag.store import build_index


def main() -> None:
    chunks = chunk_knowledge_base()
    print(f"Chunked {len(chunks)} section(s) from knowledge_base/:")
    for chunk in chunks:
        print(f"  {chunk['source']} :: {chunk['section']}")

    build_index(chunks)
    print("\nIndexed into Chroma at backend/storage/chroma/.")


if __name__ == "__main__":
    main()
