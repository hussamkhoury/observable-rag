import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExtractedRecord:
    topic: str
    column_name: str
    text: str


TEXT_COLUMNS = ("ki_text", "alt_ki_text", "bad_ki_text")


def extract(csv_path: Path) -> list[ExtractedRecord]:
    """Read a CSV file and yield non-empty text records from all text columns."""
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            topic = row.get("ki_topic", "").strip()
            if not topic:
                continue
            for col in TEXT_COLUMNS:
                text = row.get(col, "").strip()
                if text:
                    records.append(ExtractedRecord(topic=topic, column_name=col, text=text))
    return records
