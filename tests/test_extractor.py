import csv
import tempfile
from pathlib import Path

from ingest.extractor import ExtractedRecord, extract, TEXT_COLUMNS


class TestExtractor:
    def test_extracts_all_text_columns(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "ki_topic,ki_text,alt_ki_text,bad_ki_text\n"
            '"Topic A","Text A","Alt A","Bad A"\n'
            '"Topic B","Text B","Alt B","Bad B"\n'
        )

        records = extract(csv_file)

        assert len(records) == 6
        assert records[0] == ExtractedRecord(topic="Topic A", column_name="ki_text", text="Text A")
        assert records[1] == ExtractedRecord(topic="Topic A", column_name="alt_ki_text", text="Alt A")
        assert records[2] == ExtractedRecord(topic="Topic A", column_name="bad_ki_text", text="Bad A")
        assert records[3] == ExtractedRecord(topic="Topic B", column_name="ki_text", text="Text B")

    def test_skips_empty_text(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "ki_topic,ki_text,alt_ki_text,bad_ki_text\n"
            '"Topic A","Text A","","Bad A"\n'
        )

        records = extract(csv_file)

        assert len(records) == 2
        assert records[0].column_name == "ki_text"
        assert records[1].column_name == "bad_ki_text"

    def test_skips_empty_topic(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "ki_topic,ki_text,alt_ki_text,bad_ki_text\n"
            ',"Text A","Alt A","Bad A"\n'
        )

        records = extract(csv_file)

        assert len(records) == 0

    def test_handles_multiline_text(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            'ki_topic,ki_text,alt_ki_text,bad_ki_text\n'
            '"Topic A","Line 1\n\nLine 2\n\nLine 3","",""\n'
        )

        records = extract(csv_file)

        assert len(records) == 1
        assert "\n\n" in records[0].text
