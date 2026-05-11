"""Run 5 test queries against the RAG system for evaluation."""

import json
import time

import chromadb
from anthropic import Anthropic
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.types import TraceContext

load_dotenv()


QUERIES = [
    {
        "case": "clearly_in_corpus",
        "question": "How do I configure VPN access for remote workers?",
        "expected": "Should retrieve the VPN configuration article and answer with specific steps.",
    },
    {
        "case": "not_in_corpus",
        "question": "How do I train a machine learning model on AWS SageMaker?",
        "expected": "Should say 'I don't know' since ML/AWS content is not in the IT support corpus.",
    },
    {
        "case": "partially_in_corpus",
        "question": "How do I set up a secure connection and what are the best practices for remote work?",
        "expected": "Should retrieve secure connection articles but lack remote work best practices — partial answer.",
    },
    {
        "case": "ambiguous_multiple_matches",
        "question": "How do I reset a forgotten password or login?",
        "expected": "Should retrieve multiple chunks (PIN reset, username reset, VPN password, security question, etc.) — LLM must disambiguate.",
    },
    {
        "case": "adversarial",
        "question": "How do I configure VPN settings on a Cisco router?",
        "expected": "VPN is in the corpus but Cisco router config is not — should not hallucinate router steps.",
    },
]


def run_query(question: str, k: int = 4, model: str = "claude-sonnet-4-6"):
    """Run one query through the full RAG pipeline with tracing."""
    langfuse = Langfuse()
    client = Anthropic()
    chroma_client = chromadb.PersistentClient(path="data/chroma_db")
    collection = chroma_client.get_collection("knowledge_base")
    embedding_fn = collection._embedding_function

    start = time.time()

    # 1. Embed
    embed_obs = langfuse.start_observation(name="embed_query", as_type="embedding", input=question)
    query_embedding = embedding_fn([question])
    embed_obs.update(output={"embedding_dim": len(query_embedding[0])})
    embed_obs.end()
    trace_ctx = TraceContext(trace_id=embed_obs.trace_id)

    # 2. Retrieve
    retrieve_obs = langfuse.start_observation(
        trace_context=trace_ctx, name="retrieve", as_type="retriever",
        input={"question": question, "k": k},
    )
    results = collection.query(query_embeddings=query_embedding, n_results=k)
    chunks = [
        {
            "text": doc,
            "topic": meta["topic"],
            "source_column": meta["source_column"],
            "distance": dist,
        }
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0],
        )
    ]
    retrieve_obs.update(output={"chunks_retrieved": len(chunks)})
    retrieve_obs.end()

    # 3. Build prompt
    context_parts = []
    for i, c in enumerate(chunks):
        context_parts.append(f"[{i+1}] (topic: {c['topic']}, source: {c['source_column']}, distance: {c['distance']:.4f})\n{c['text']}")
    context_block = "\n\n---\n\n".join(context_parts)
    prompt = (
        "Answer the question using only the context below.\n"
        "If the context doesn't contain the answer, say \"I don't know.\"\n"
        "\nContext:\n"
        f"{context_block}\n"
        "\n"
        f"Question: {question}"
    )
    build_obs = langfuse.start_observation(
        trace_context=trace_ctx, name="build_prompt", as_type="span",
        input={"question": question, "chunks": len(chunks)},
    )
    build_obs.update(output={"prompt_length": len(prompt)})
    build_obs.end()

    # 4. LLM call
    llm_obs = langfuse.start_observation(
        trace_context=trace_ctx, name="llm_call", as_type="generation",
        input=prompt, model=model,
    )
    response = client.messages.create(model=model, max_tokens=1024, messages=[{"role": "user", "content": prompt}])
    answer = response.content[0].text
    usage = response.usage
    llm_obs.update(
        output=answer,
        usage_details={"input": usage.input_tokens, "output": usage.output_tokens},
    )
    llm_obs.end()

    latency = time.time() - start
    langfuse.flush()

    return {
        "answer": answer,
        "chunks": chunks,
        "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens},
        "latency_s": round(latency, 2),
    }


def main():
    for q in QUERIES:
        print(f"\n{'='*60}")
        print(f"CASE: {q['case']}")
        print(f"QUESTION: {q['question']}")
        print(f"EXPECTED: {q['expected']}")
        print(f"{'='*60}")

        result = run_query(q["question"])

        print(f"\nRETRIEVED CHUNKS ({len(result['chunks'])}):")
        for i, c in enumerate(result["chunks"]):
            print(f"  [{i+1}] topic=\"{c['topic']}\" | source={c['source_column']} | distance={c['distance']:.4f}")
            print(f"      {c['text'][:120]}...")

        print(f"\nLLM ANSWER:")
        print(f"  {result['answer']}")
        print(f"\nMETRICS:")
        print(f"  Input tokens:  {result['usage']['input_tokens']}")
        print(f"  Output tokens: {result['usage']['output_tokens']}")
        print(f"  Total tokens:  {result['usage']['input_tokens'] + result['usage']['output_tokens']}")
        print(f"  Latency:      {result['latency_s']}s")
        print(f"\n  Trace URL: check Langfuse at http://localhost:3000")


if __name__ == "__main__":
    main()
