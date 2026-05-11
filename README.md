# observable-rag

## Data Source

Synthetic IT-related knowledge items used for chunking and retrieval:

- [Synthetic IT-Related Knowledge Items](https://www.kaggle.com/datasets/dkhundley/synthetic-it-related-knowledge-items) — 100 knowledge items covering IT support topics (email setup, VPN configuration, software troubleshooting, etc.)
- Each item contains a topic, a primary text (`ki_text`), an alternative phrasing (`alt_ki_text`), and an incorrect version (`bad_ki_text`)
- Stored in `data/synthetic_knowledge_items.csv`

## Ingestion

The ingestion pipeline reads the CSV, chunks each text column at semantic boundaries (paragraphs, headings) into ~500-token pieces, and outputs `data/chunks.jsonl`.

```bash
uv run python -c "from ingest.pipeline import run; from pathlib import Path; run(Path('data/synthetic_knowledge_items.csv'), Path('data/chunks.jsonl'))"
```

See `data/chunking_description.txt` for details on the chunking approach.
