FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

COPY . .

RUN uv run python -c "from ingest.pipeline import run; from pathlib import Path; run(Path('data/synthetic_knowledge_items.csv'), Path('data/chunks.jsonl'))" \
    && uv run python -c "from ingest.embedder import embed; from pathlib import Path; embed(Path('data/chunks.jsonl'), persist_dir=Path('data/chroma_db'))"

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
