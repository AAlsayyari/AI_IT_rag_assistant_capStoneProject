#!/usr/bin/env python3
"""
test_query.py — CLI Retrieval Test for Arabic IT/PC Troubleshooting RAG
========================================================================
Usage:
    python test_query.py "حل مشكلة الشاشة الزرقاء"
    python test_query.py "الجهاز لا يقلع"
    python test_query.py "how to fix blue screen"
"""

import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "arabic_it_troubleshooting"
EMBED_MODEL_NAME = "BAAI/bge-m3"
TOP_K = 5


def query_rag(question: str, top_k: int = TOP_K) -> None:
    """Embed the question, query ChromaDB, and print results."""

    # ── Validate ChromaDB exists ──
    if not CHROMA_DIR.exists():
        print(f"❌ ChromaDB not found at {CHROMA_DIR}")
        print("   Run 'python ingest.py' first to build the vector database.")
        sys.exit(1)

    # ── Load embedding model ──
    print(f"🧠 Loading embedding model: {EMBED_MODEL_NAME}")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # ── Connect to ChromaDB ──
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        print(f"❌ Collection '{COLLECTION_NAME}' not found in ChromaDB.")
        print("   Run 'python ingest.py' first.")
        sys.exit(1)

    doc_count = collection.count()
    print(f"📚 Collection '{COLLECTION_NAME}' has {doc_count} chunks.\n")

    # ── Embed the query ──
    query_embedding = embed_model.encode(
        question, normalize_embeddings=True
    ).tolist()

    # ── Query ──
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, doc_count),
        include=["documents", "metadatas", "distances"],
    )

    # ── Display results ──
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print("=" * 70)
    print(f"  🔎 Query: {question}")
    print(f"  📊 Top {len(documents)} results (cosine distance — lower = better)")
    print("=" * 70)

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), 1):
        similarity = 1.0 - dist  # cosine similarity = 1 - cosine distance
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        header = meta.get("header", "")

        # Truncate content for display
        snippet = doc[:400].replace("\n", " ↵ ")
        if len(doc) > 400:
            snippet += " …"

        print(f"\n{'─' * 70}")
        print(f"  Result #{i}")
        print(f"  📄 Source : {source}")
        print(f"  📑 Page   : {page}")
        if header:
            print(f"  📌 Header : {header}")
        print(f"  📏 Score  : {similarity:.4f} (similarity) | {dist:.4f} (distance)")
        print(f"  📝 Content:")
        print(f"     {snippet}")

    print(f"\n{'─' * 70}")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_query.py \"<your Arabic question>\"")
        print()
        print("Examples:")
        print('  python test_query.py "حل مشكلة الشاشة الزرقاء"')
        print('  python test_query.py "الجهاز لا يقلع"')
        print('  python test_query.py "مشكلة في الواي فاي"')
        print('  python test_query.py "الكمبيوتر بطيء جدا"')
        sys.exit(0)

    question = " ".join(sys.argv[1:])
    query_rag(question)


if __name__ == "__main__":
    main()
