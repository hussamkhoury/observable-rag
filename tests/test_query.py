from unittest.mock import MagicMock, patch

import pytest

from ingest.query import query, QueryResult, _build_context, _build_prompt


@pytest.fixture
def mock_deps():
    """Mock all external dependencies: Langfuse, Anthropic, ChromaDB."""
    with (
        patch("ingest.query.Langfuse") as mock_langfuse_cls,
        patch("ingest.query.Anthropic") as mock_anthropic_cls,
        patch("ingest.query.chromadb.PersistentClient") as mock_chroma_cls,
    ):
        mock_langfuse = MagicMock()
        mock_langfuse_cls.return_value = mock_langfuse

        def make_obs(**kwargs):
            obs = MagicMock()
            obs.trace_id = "trace-123"
            obs.update = MagicMock()
            obs.end = MagicMock()
            return obs

        mock_langfuse.start_observation = MagicMock(side_effect=make_obs)

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Use a VPN client to connect.")]
        mock_client.messages.create.return_value = mock_response

        mock_collection = MagicMock()
        mock_collection._embedding_function = MagicMock(return_value=[[0.1, 0.2, 0.3]])
        mock_collection.query.return_value = {
            "documents": [["Chunk about VPN setup."]],
            "metadatas": [[{"topic": "VPN Setup", "source_column": "ki_text", "token_count": 50}]],
            "distances": [[0.15]],
        }
        mock_chroma_client = MagicMock()
        mock_chroma_client.get_collection.return_value = mock_collection
        mock_chroma_cls.return_value = mock_chroma_client

        yield {
            "langfuse": mock_langfuse,
            "anthropic": mock_client,
            "collection": mock_collection,
        }


class TestQuery:
    def test_returns_answer_and_sources(self, mock_deps):
        result = query("How do I set up VPN?")
        assert isinstance(result, QueryResult)
        assert "VPN" in result.answer
        assert len(result.sources) == 1
        assert result.sources[0]["topic"] == "VPN Setup"

    def test_creates_four_observations(self, mock_deps):
        query("Test?")
        assert mock_deps["langfuse"].start_observation.call_count == 4

    def test_observation_names(self, mock_deps):
        query("Test?")
        names = [c.kwargs["name"] for c in mock_deps["langfuse"].start_observation.call_args_list]
        assert names == ["embed_query", "retrieve", "build_prompt", "llm_call"]

    def test_observation_types(self, mock_deps):
        query("Test?")
        types = [c.kwargs["as_type"] for c in mock_deps["langfuse"].start_observation.call_args_list]
        assert types == ["embedding", "retriever", "span", "generation"]

    def test_child_observations_share_trace_context(self, mock_deps):
        query("Test?")
        calls = mock_deps["langfuse"].start_observation.call_args_list
        for call in calls[1:]:
            tc = call.kwargs["trace_context"]
            if hasattr(tc, "trace_id"):
                assert tc.trace_id == "trace-123"
            else:
                assert tc["trace_id"] == "trace-123"

    def test_calls_llm_with_prompt(self, mock_deps):
        query("What is email?")
        call_kwargs = mock_deps["anthropic"].messages.create.call_args
        prompt = call_kwargs.kwargs["messages"][0]["content"]
        assert "What is email?" in prompt
        assert "Context:" in prompt

    def test_queries_chroma_with_embedding(self, mock_deps):
        query("Test?", k=3)
        call_kwargs = mock_deps["collection"].query.call_args
        assert call_kwargs.kwargs["n_results"] == 3

    def test_uses_custom_model(self, mock_deps):
        query("Test?", model="claude-opus-4-6")
        call_kwargs = mock_deps["anthropic"].messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-opus-4-6"

    def test_llm_generation_has_model(self, mock_deps):
        query("Test?")
        llm_call = mock_deps["langfuse"].start_observation.call_args_list[3]
        assert llm_call.kwargs["model"] == "claude-sonnet-4-6"

    def test_flushes_langfuse(self, mock_deps):
        query("Test?")
        mock_deps["langfuse"].flush.assert_called_once()

    def test_i_dont_know_in_prompt(self, mock_deps):
        mock_deps["collection"].query.return_value = {
            "documents": [["Unrelated cooking content."]],
            "metadatas": [[{"topic": "Cooking", "source_column": "ki_text", "token_count": 30}]],
            "distances": [[0.9]],
        }
        query("How do I configure VPN?")
        prompt = mock_deps["anthropic"].messages.create.call_args.kwargs["messages"][0]["content"]
        assert "I don't know" in prompt

    def test_all_observations_ended(self, mock_deps):
        query("Test?")
        for obs in [c.return_value for c in mock_deps["langfuse"].start_observation.call_args_list]:
            obs.end.assert_called_once()


class TestBuildContext:
    def test_build_context(self):
        from ingest.query import RetrievedChunk
        chunks = [
            RetrievedChunk(text="VPN setup steps.", topic="VPN", source_column="ki_text", distance=0.1),
            RetrievedChunk(text="Email config.", topic="Email", source_column="alt_ki_text", distance=0.2),
        ]
        context, sources = _build_context(chunks)
        assert "VPN setup steps." in context
        assert "Email config." in context
        assert sources == [{"topic": "VPN", "source_column": "ki_text"}, {"topic": "Email", "source_column": "alt_ki_text"}]

    def test_build_prompt(self):
        prompt = _build_prompt("What is VPN?", "[1] VPN info here.")
        assert "What is VPN?" in prompt
        assert "I don't know" in prompt
        assert "[1] VPN info here." in prompt
