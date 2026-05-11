from dataclasses import dataclass

import chromadb
from anthropic import Anthropic
from langfuse import Langfuse


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    topic: str
    source_column: str
    distance: float


@dataclass(frozen=True)
class QueryResult:
    answer: str
    sources: list[dict]


def _build_context(chunks: list[RetrievedChunk]) -> tuple[str, list[dict]]:
    """Build the context block and sources list from retrieved chunks."""
    context_parts = []
    sources = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[{i+1}] (topic: {chunk.topic}, source: {chunk.source_column})\n{chunk.text}"
        )
        sources.append({"topic": chunk.topic, "source_column": chunk.source_column})
    return "\n\n---\n\n".join(context_parts), sources


def _build_prompt(question: str, context_block: str) -> str:
    return (
        "Answer the question using only the context below.\n"
        "If the context doesn't contain the answer, say \"I don't know.\"\n"
        "\n"
        "Context:\n"
        f"{context_block}\n"
        "\n"
        f"Question: {question}"
    )


def query(
    question: str,
    *,
    collection_name: str = "knowledge_base",
    persist_dir: str = "data/chroma_db",
    k: int = 4,
    model: str = "claude-sonnet-4-6",
    trace_name: str = "ask",
) -> QueryResult:
    """Full RAG pipeline: embed question → retrieve → build prompt → LLM call.

    Creates a single Langfuse trace with child observations for each step.
    """
    from langfuse.types import TraceContext

    langfuse = Langfuse()
    client = Anthropic()
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    collection = chroma_client.get_collection(collection_name)
    embedding_fn = collection._embedding_function

    # 1. Embed the question
    embed_obs = langfuse.start_observation(
        name="embed_query",
        as_type="embedding",
        input=question,
    )
    query_embedding = embedding_fn([question])
    embed_obs.update(output={"embedding_dim": len(query_embedding[0])})
    embed_obs.end()
    trace_ctx = TraceContext(trace_id=embed_obs.trace_id)

    # 2. Retrieve top-K chunks
    retrieve_obs = langfuse.start_observation(
        trace_context=trace_ctx,
        name="retrieve",
        as_type="retriever",
        input={"question": question, "k": k},
    )
    results = collection.query(query_embeddings=query_embedding, n_results=k)
    chunks = [
        RetrievedChunk(
            text=doc, topic=meta["topic"], source_column=meta["source_column"], distance=dist,
        )
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0],
        )
    ]
    retrieve_obs.update(output={"chunks_retrieved": len(chunks)})
    retrieve_obs.end()

    # 3. Build prompt
    context_block, sources = _build_context(chunks)
    prompt = _build_prompt(question, context_block)
    build_obs = langfuse.start_observation(
        trace_context=trace_ctx,
        name="build_prompt",
        as_type="span",
        input={"question": question, "chunks": len(chunks)},
    )
    build_obs.update(output={"prompt_length": len(prompt)})
    build_obs.end()

    # 4. LLM call
    llm_obs = langfuse.start_observation(
        trace_context=trace_ctx,
        name="llm_call",
        as_type="generation",
        input=prompt,
        model=model,
    )
    response = client.messages.create(
        model=model, max_tokens=1024, messages=[{"role": "user", "content": prompt}],
    )
    answer = response.content[0].text
    llm_obs.update(output=answer)
    llm_obs.end()

    langfuse.flush()
    return QueryResult(answer=answer, sources=sources)
