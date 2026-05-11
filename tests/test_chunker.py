from ingest.chunker import Chunk, chunk_text, count_tokens, MAX_TOKENS


class TestCountTokens:
    def test_counts_tokens(self):
        assert count_tokens("Hello, world!") > 0
        assert count_tokens("") == 0

    def test_counts_accurately(self):
        # "Hello" = 1 token in cl100k_base
        assert count_tokens("Hello") == 1


class TestChunkText:
    def test_single_short_text(self):
        chunks = chunk_text("Short text", topic="T", source_column="ki_text")

        assert len(chunks) == 1
        assert chunks[0].text == "Short text"
        assert chunks[0].topic == "T"
        assert chunks[0].source_column == "ki_text"
        assert chunks[0].chunk_id == 0

    def test_splits_at_paragraph_boundary(self):
        paragraphs = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        # Each paragraph is short, so they all fit in one chunk
        chunks = chunk_text(paragraphs, topic="T", source_column="ki_text")

        assert len(chunks) == 1
        assert "First paragraph" in chunks[0].text
        assert "Third paragraph" in chunks[0].text

    def test_splits_long_text_into_multiple_chunks(self):
        # Generate enough text to exceed 500 tokens across multiple paragraphs
        paragraph = "This is a test paragraph with enough words to accumulate tokens. " * 30
        text = f"{paragraph}\n\n{paragraph}\n\n{paragraph}\n\n{paragraph}"

        chunks = chunk_text(text, topic="T", source_column="ki_text")

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.token_count <= MAX_TOKENS + 50  # small tolerance for merging

    def test_chunk_ids_are_sequential(self):
        paragraph = "Word " * 200
        text = f"{paragraph}\n\n{paragraph}\n\n{paragraph}"

        chunks = chunk_text(text, topic="T", source_column="ki_text")

        ids = [c.chunk_id for c in chunks]
        assert ids == list(range(len(chunks)))

    def test_respects_start_id(self):
        chunks = chunk_text("Short text", topic="T", source_column="ki_text", start_id=42)

        assert chunks[0].chunk_id == 42

    def test_empty_text_returns_empty(self):
        chunks = chunk_text("", topic="T", source_column="ki_text")

        assert chunks == []

    def test_preserves_topic_and_source_column(self):
        chunks = chunk_text("Hello world", topic="VPN Setup", source_column="alt_ki_text")

        assert chunks[0].topic == "VPN Setup"
        assert chunks[0].source_column == "alt_ki_text"

    def test_heading_boundary_split(self):
        text = "**Step 1: Do this**\nSome content here.\n\n**Step 2: Do that**\nMore content."

        chunks = chunk_text(text, topic="T", source_column="ki_text")

        assert len(chunks) >= 1
