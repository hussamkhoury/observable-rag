import re
from dataclasses import dataclass

import tiktoken


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    topic: str
    source_column: str
    text: str
    token_count: int


_ENC = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 500
_BOUNDARY_RE = re.compile(r"\n{2,}|\n(?=\*\*)")


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _split_boundaries(text: str) -> list[str]:
    """Split text at paragraph breaks and heading boundaries."""
    parts = _BOUNDARY_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, topic: str, source_column: str, start_id: int = 0) -> list[Chunk]:
    """Split text into ~500-token chunks at semantic boundaries.

    Splits on paragraph breaks and heading boundaries, then greedily
    merges small segments up to MAX_TOKENS.
    """
    segments = _split_boundaries(text)
    if not segments:
        return []

    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_tokens = 0
    chunk_id = start_id

    for segment in segments:
        seg_tokens = count_tokens(segment)

        if current_tokens + seg_tokens > MAX_TOKENS and current_parts:
            merged_text = "\n\n".join(current_parts)
            chunks.append(Chunk(
                chunk_id=chunk_id,
                topic=topic,
                source_column=source_column,
                text=merged_text,
                token_count=count_tokens(merged_text),
            ))
            chunk_id += 1
            current_parts = []
            current_tokens = 0

        current_parts.append(segment)
        current_tokens += seg_tokens

    if current_parts:
        merged_text = "\n\n".join(current_parts)
        chunks.append(Chunk(
            chunk_id=chunk_id,
            topic=topic,
            source_column=source_column,
            text=merged_text,
            token_count=count_tokens(merged_text),
        ))

    return chunks
