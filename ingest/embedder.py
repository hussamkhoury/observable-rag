import json
from pathlib import Path

import chromadb


def embed(chunks_path: Path, collection_name: str = "knowledge_base", persist_dir: Path | None = None) -> chromadb.Collection:
    """Embed chunks from a JSONL file into a ChromaDB collection.

    Uses OpenAI's text-embedding-3-small model via ChromaDB's built-in support.
    """
    client_settings = chromadb.Settings(anonymized_telemetry=False)
    if persist_dir:
        client = chromadb.PersistentClient(path=str(persist_dir), settings=client_settings)
    else:
        client = chromadb.Client(settings=client_settings)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            ids=[str(c["chunk_id"]) for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{
                "topic": c["topic"],
                "source_column": c["source_column"],
                "token_count": c["token_count"],
            } for c in batch],
        )

    return collection
