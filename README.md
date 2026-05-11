# observable-rag

A RAG (Retrieval-Augmented Generation) system that answers IT support questions over a corpus of 100 synthetic knowledge items. Every step of the pipeline — embedding, retrieval, prompt construction, and LLM generation — is traced as a child span in [Langfuse](https://langfuse.com), giving you full observability into what your RAG system is doing and why it produces the answers it does.

## Quickstart

```bash
cp .env.example .env          # Add your ANTHROPIC_API_KEY and Langfuse keys
docker compose up --build     # Builds app, starts Langfuse on :3000
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I configure VPN access?"}'
```

Open `http://localhost:3000` to see Langfuse traces for every query.

## Architecture

```
                        ┌─────────────────────────────────┐
                        │          FastAPI (:8000)          │
                        │             POST /ask             │
                        └──────────────┬──────────────────┘
                                       │
                        ┌──────────────▼──────────────────┐
                        │         RAG Pipeline            │
                        │                                 │
                        │  1. embed_query   (embedding)   │
                        │         │                       │
                        │  2. retrieve     (retriever)    │
                        │         │                       │
                        │  3. build_prompt (span)         │
                        │         │                       │
                        │  4. llm_call     (generation)   │
                        └────┬──────────┬─────────────────┘
                             │          │
                  ┌──────────▼──┐  ┌────▼───────┐
                  │  ChromaDB   │  │  Anthropic  │
                  │  (vectors)  │  │  Claude API │
                  └─────────────┘  └─────────────┘
                             │
                  ┌──────────▼──────────────────┐
                  │         Langfuse (:3000)     │
                  │   1 trace per query           │
                  │   4 child observations        │
                  └──────────────────────────────┘
```

Each `/ask` request creates one Langfuse trace with four child observations linked via `TraceContext`. The trace shows embedding dimensions, retrieval distances, prompt length, token usage, and the final LLM output.

## Data Source

[Synthetic IT-Related Knowledge Items](https://www.kaggle.com/datasets/dkhundley/synthetic-it-related-knowledge-items) — 100 knowledge items covering IT support topics (email setup, VPN configuration, software troubleshooting, etc.). Each item contains a topic, a primary text (`ki_text`), an alternative phrasing (`alt_ki_text`), and an incorrect version (`bad_ki_text`). Stored in `data/synthetic_knowledge_items.csv`.

## Ingestion

The ingestion pipeline reads the CSV, chunks each text column at semantic boundaries (paragraphs, headings) into ~500-token pieces, and outputs `data/chunks.jsonl`.

```bash
uv run python -c "from ingest.pipeline import run; from pathlib import Path; run(Path('data/synthetic_knowledge_items.csv'), Path('data/chunks.jsonl'))"
```

See `data/chunking_description.txt` for details on the chunking approach.

## Evaluation

Five test queries were run against the RAG system. Results below.

### Test Queries

| Case | Question | Retrieved Relevant? | Answer Quality | Hallucination? |
|------|----------|:---:|:---:|:---:|
| Clearly in corpus | How do I configure VPN access for remote workers? | Yes — all 4 chunks are VPN config | Detailed answer with steps from general and Widgetco sources | No |
| Not in corpus | How do I train a machine learning model on AWS SageMaker? | No — retrieved VPN/Remote Desktop/IT Request (distance ~0.75) | "I don't know." | No |
| Partially in corpus | How do I set up a secure connection and best practices for remote work? | Yes — VPN, database, server connections | Answered secure setup, but "best practices" includes details not fully in chunks | Minor |
| Ambiguous | How do I reset a forgotten password or login? | Yes — Network Password, Username, Computer Password | Disambiguated into password reset and username reset | No |
| Adversarial | How do I configure VPN settings on a Cisco router? | Partial — VPN chunks but no Cisco router content | "I don't know." | No |

### Key Findings

- **"I don't know" guard works** — correctly triggered for out-of-corpus (distance ~0.75) and adversarial queries
- **Retrieval quality correlates with distance** — exact match at 0.15, irrelevant at 0.75
- **Partial corpus case had minor hallucination** — the "best practices" section included general advice not grounded in retrieved chunks (VPN usage, contact IT)
- **Ambiguous query was handled well** — LLM disambiguated across multiple password/login reset articles
- **LLM stayed grounded in retrieved chunks** for 4 of 5 cases

### Metrics

| Case | Input Tokens | Output Tokens | Total Tokens | Latency |
|------|:---:|:---:|:---:|:---:|
| Clearly in corpus | 1,647 | 454 | 2,101 | 9.63s |
| Not in corpus | 2,041 | 6 | 2,047 | 2.72s |
| Partially in corpus | 1,600 | 337 | 1,937 | 6.28s |
| Ambiguous | 1,778 | 373 | 2,151 | 5.12s |
| Adversarial | 1,986 | 6 | 1,992 | 2.38s |

### Notes

- The `bad_ki_text` column contains intentionally incorrect information — the LLM correctly ignored or downweighted these chunks in most cases, but the partially-in-corpus query did reference bad_ki_text chunks for database/server connections
- Top-K=4 retrieval was used for all queries
- Model: claude-sonnet-4-6
- Embedding: ChromaDB default (all-MiniLM-L6-v2)

## Langfuse Trace Example

![Langfuse trace showing 4 child observations for a single RAG query](docs/images/langfuse-trace.png)

Each trace shows the full pipeline: embedding dimensions, retrieval distances per chunk, prompt construction length, and the LLM generation with token usage.

## What I Learned

**Observability is not optional for RAG.** Without tracing, a wrong answer is a mystery. With Langfuse, you can see exactly which chunks were retrieved, their distance scores, and whether the LLM grounded its response in context or drifted. The "partial hallucination" in our eval would have been invisible without this.

**"I don't know" is a feature, not a bug.** The simplest guard — instructing the LLM to say "I don't know" when context doesn't contain the answer — eliminated hallucinations on out-of-corpus and adversarial queries. The retrieval distances confirm this: irrelevant chunks scored ~0.75 vs ~0.15 for relevant ones. A distance threshold could add a second guardrail.

**Chunking strategy matters more than I expected.** Semantic boundary splitting (paragraphs and headings) with greedy merging to 500 tokens produced clean, coherent chunks. But the `bad_ki_text` column — intentionally incorrect content — still got retrieved and almost influenced one answer. Garbage in, garbage out applies to retrieval just as much as generation.

**Tracing teaches you about your system.** The 4-observation pattern (embed → retrieve → build_prompt → llm_call) isn't just for debugging. It gives you a performance profile: where does latency come from? How many tokens does the prompt consume? Are retrieval distances reasonable? These questions are unanswerable without the trace data.

**Keep it modular.** Splitting the pipeline into extract → chunk → embed → query made each step independently testable and debuggable. The gateway abstraction (Anthropic/OpenAI) means switching models is a config change, not a rewrite.
