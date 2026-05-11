import json
from pathlib import Path

import pytest

from ingest.embedder import embed


class TestEmbedder:
    def test_embeds_chunks_into_collection(self, tmp_path):
        chunks_file = tmp_path / "chunks.jsonl"
        chunks_file.write_text(
            json.dumps({"chunk_id": 0, "topic": "VPN", "source_column": "ki_text", "text": "Configure VPN access on your device.", "token_count": 8}) + "\n"
            + json.dumps({"chunk_id": 1, "topic": "Email", "source_column": "ki_text", "text": "Set up company email on mobile.", "token_count": 7}) + "\n"
        )

        persist_dir = tmp_path / "chroma_db"
        collection = embed(chunks_file, collection_name="test_kb", persist_dir=persist_dir)

        assert collection.count() == 2

    def test_collection_uses_cosine_similarity(self, tmp_path):
        chunks_file = tmp_path / "chunks.jsonl"
        chunks_file.write_text(
            json.dumps({"chunk_id": 0, "topic": "T", "source_column": "ki_text", "text": "Hello world", "token_count": 2}) + "\n"
        )

        persist_dir = tmp_path / "chroma_db"
        collection = embed(chunks_file, collection_name="test_cosine", persist_dir=persist_dir)

        assert collection.metadata["hnsw:space"] == "cosine"

    def test_metadata_preserved(self, tmp_path):
        chunks_file = tmp_path / "chunks.jsonl"
        chunks_file.write_text(
            json.dumps({"chunk_id": 0, "topic": "VPN Setup", "source_column": "alt_ki_text", "text": "Install VPN client.", "token_count": 4}) + "\n"
        )

        persist_dir = tmp_path / "chroma_db"
        collection = embed(chunks_file, collection_name="test_meta", persist_dir=persist_dir)

        result = collection.get(ids=["0"])
        assert result["metadatas"][0]["topic"] == "VPN Setup"
        assert result["metadatas"][0]["source_column"] == "alt_ki_text"
        assert result["metadatas"][0]["token_count"] == 4

    def test_persists_to_disk(self, tmp_path):
        chunks_file = tmp_path / "chunks.jsonl"
        chunks_file.write_text(
            json.dumps({"chunk_id": 0, "topic": "T", "source_column": "ki_text", "text": "Persistent data.", "token_count": 3}) + "\n"
        )

        persist_dir = tmp_path / "chroma_db"
        embed(chunks_file, collection_name="test_persist", persist_dir=persist_dir)

        # Reset the ChromaDB client singleton so we can reopen with the same path
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient._identifier_to_system = {}

        import chromadb
        client = chromadb.PersistentClient(path=str(persist_dir))
        collection = client.get_collection("test_persist")
        assert collection.count() == 1
