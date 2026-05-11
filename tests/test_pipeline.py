import json
from pathlib import Path

from ingest.pipeline import run


class TestPipeline:
    def test_produces_jsonl_output(self, tmp_path):
        csv_file = tmp_path / "input.csv"
        csv_file.write_text(
            "ki_topic,ki_text,alt_ki_text,bad_ki_text\n"
            '"Topic A","Short text one.","Alt text one.",""\n'
        )

        output_file = tmp_path / "output.jsonl"
        chunks = run(csv_file, output_file)

        assert len(chunks) == 2
        assert output_file.exists()

        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 2

        first = json.loads(lines[0])
        assert first["topic"] == "Topic A"
        assert first["source_column"] == "ki_text"
        assert first["text"] == "Short text one."
        assert "chunk_id" in first
        assert "token_count" in first

    def test_creates_output_directory_if_missing(self, tmp_path):
        csv_file = tmp_path / "input.csv"
        csv_file.write_text(
            "ki_topic,ki_text,alt_ki_text,bad_ki_text\n"
            '"Topic","Hello","World",""\n'
        )

        output_file = tmp_path / "subdir" / "output.jsonl"
        run(csv_file, output_file)

        assert output_file.exists()

    def test_chunk_ids_are_globally_unique(self, tmp_path):
        csv_file = tmp_path / "input.csv"
        csv_file.write_text(
            "ki_topic,ki_text,alt_ki_text,bad_ki_text\n"
            '"Topic A","Text A","Alt A",""\n'
            '"Topic B","Text B","Alt B",""\n'
        )

        output_file = tmp_path / "output.jsonl"
        chunks = run(csv_file, output_file)

        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_returns_all_chunks(self, tmp_path):
        csv_file = tmp_path / "input.csv"
        csv_file.write_text(
            "ki_topic,ki_text,alt_ki_text,bad_ki_text\n"
            '"Topic","Text","Alt","Bad"\n'
        )

        output_file = tmp_path / "output.jsonl"
        chunks = run(csv_file, output_file)

        lines = output_file.read_text().strip().split("\n")
        assert len(chunks) == len(lines)
