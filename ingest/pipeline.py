import json
from pathlib import Path

from ingest.chunker import Chunk, chunk_text
from ingest.extractor import ExtractedRecord, extract


def run(csv_path: Path, output_path: Path) -> list[Chunk]:
    """Run the full ingestion pipeline: extract → chunk → write JSONL."""
    records = extract(csv_path)
    all_chunks: list[Chunk] = []
    chunk_id = 0

    for record in records:
        chunks = chunk_text(record.text, record.topic, record.column_name, start_id=chunk_id)
        all_chunks.extend(chunks)
        chunk_id += len(chunks)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps({
                "chunk_id": chunk.chunk_id,
                "topic": chunk.topic,
                "source_column": chunk.source_column,
                "text": chunk.text,
                "token_count": chunk.token_count,
            }) + "\n")

    return all_chunks
