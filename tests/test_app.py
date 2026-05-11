from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


class TestAskEndpoint:
    @patch("app.rag_query")
    def test_ask_returns_answer_and_sources(self, mock_query):
        mock_query.return_value = MagicMock(answer="Use VPN.", sources=[{"topic": "VPN"}])

        response = client.post("/ask", json={"question": "How do I set up VPN?"})

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Use VPN."
        assert data["sources"] == [{"topic": "VPN"}]

    @patch("app.rag_query")
    def test_ask_passes_k_and_model(self, mock_query):
        mock_query.return_value = MagicMock(answer="Yes", sources=[])

        client.post("/ask", json={"question": "Test?", "k": 5, "model": "claude-opus-4-6"})

        mock_query.assert_called_once()
        assert mock_query.call_args.kwargs["k"] == 5
        assert mock_query.call_args.kwargs["model"] == "claude-opus-4-6"

    @patch("app.rag_query")
    def test_ask_defaults(self, mock_query):
        mock_query.return_value = MagicMock(answer="Default", sources=[])

        client.post("/ask", json={"question": "Hello?"})

        mock_query.assert_called_once()
        assert mock_query.call_args.kwargs["k"] == 4
        assert mock_query.call_args.kwargs["model"] == "claude-sonnet-4-6"
