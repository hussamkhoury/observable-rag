from unittest.mock import MagicMock, patch

import pytest

from gateways.anthropic_gateway import AnthropicGateway


@pytest.fixture
def mock_anthropic_client():
    with patch("gateways.anthropic_gateway.Anthropic") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_instance


class TestAnthropicGatewayInit:
    def test_uses_provided_api_key(self, mock_anthropic_client):
        AnthropicGateway(api_key="test-key-123")

        from gateways.anthropic_gateway import Anthropic
        Anthropic.assert_called_once_with(api_key="test-key-123")

    def test_falls_back_to_env_var_when_no_key_provided(self, mock_anthropic_client):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key-456"}, clear=False):
            AnthropicGateway()

        from gateways.anthropic_gateway import Anthropic
        Anthropic.assert_called_once_with(api_key="env-key-456")


class TestAnthropicGatewaySendMessage:
    def test_send_message_returns_response_text(self, mock_anthropic_client):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello from Claude")]
        mock_anthropic_client.messages.create.return_value = mock_response

        gateway = AnthropicGateway(api_key="test-key")
        result = gateway.send_message("Hello, Claude")

        assert result == "Hello from Claude"
        mock_anthropic_client.messages.create.assert_called_once_with(
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello, Claude"}],
            model="claude-opus-4-6",
        )

    def test_send_message_uses_custom_model(self, mock_anthropic_client):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hi")]
        mock_anthropic_client.messages.create.return_value = mock_response

        gateway = AnthropicGateway(api_key="test-key")
        gateway.send_message("Hello", model="claude-sonnet-4-6")

        mock_anthropic_client.messages.create.assert_called_once()
        call_kwargs = mock_anthropic_client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-sonnet-4-6"

    def test_send_message_uses_custom_max_tokens(self, mock_anthropic_client):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hi")]
        mock_anthropic_client.messages.create.return_value = mock_response

        gateway = AnthropicGateway(api_key="test-key")
        gateway.send_message("Hello", max_tokens=2048)

        call_kwargs = mock_anthropic_client.messages.create.call_args
        assert call_kwargs.kwargs["max_tokens"] == 2048


class TestAnthropicGatewayStreamMessage:
    def test_stream_message_yields_text_deltas(self, mock_anthropic_client):
        mock_stream = MagicMock()
        mock_stream.text_stream.__iter__ = MagicMock(return_value=iter(["Hello", " from", " Claude"]))
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_anthropic_client.messages.stream.return_value = mock_stream

        gateway = AnthropicGateway(api_key="test-key")
        result = list(gateway.stream_message("Hello, Claude"))

        assert result == ["Hello", " from", " Claude"]
        mock_anthropic_client.messages.stream.assert_called_once_with(
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello, Claude"}],
            model="claude-opus-4-6",
        )

    def test_stream_message_uses_custom_model(self, mock_anthropic_client):
        mock_stream = MagicMock()
        mock_stream.text_stream.__iter__ = MagicMock(return_value=iter(["Hi"]))
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_anthropic_client.messages.stream.return_value = mock_stream

        gateway = AnthropicGateway(api_key="test-key")
        list(gateway.stream_message("Hello", model="claude-sonnet-4-6"))

        call_kwargs = mock_anthropic_client.messages.stream.call_args
        assert call_kwargs.kwargs["model"] == "claude-sonnet-4-6"


class TestAnthropicGatewaySendMessageTraced:
    def test_send_message_traced_returns_response_text(self, mock_anthropic_client):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="2")]
        mock_anthropic_client.messages.create.return_value = mock_response

        mock_langfuse = MagicMock()
        mock_observation = MagicMock()
        mock_observation.__enter__ = MagicMock(return_value=None)
        mock_observation.__exit__ = MagicMock(return_value=False)
        mock_langfuse.start_as_current_observation.return_value = mock_observation

        gateway = AnthropicGateway(api_key="test-key")

        with patch("langfuse.get_client", return_value=mock_langfuse):
            result = gateway.send_message_traced(
                "1 + 1 = ",
                name="test-chat",
                metadata={"source": "calculator"},
            )

        assert result == "2"
        mock_langfuse.start_as_current_observation.assert_called_once_with(
            as_type="generation",
            name="test-chat",
            metadata={"source": "calculator"},
            model="claude-opus-4-6",
            input=[{"role": "user", "content": "1 + 1 = "}],
        )
        mock_langfuse.update_current_generation.assert_called_once_with(output="2")

    def test_send_message_traced_with_system_prompt(self, mock_anthropic_client):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="2")]
        mock_anthropic_client.messages.create.return_value = mock_response

        mock_langfuse = MagicMock()
        mock_observation = MagicMock()
        mock_observation.__enter__ = MagicMock(return_value=None)
        mock_observation.__exit__ = MagicMock(return_value=False)
        mock_langfuse.start_as_current_observation.return_value = mock_observation

        gateway = AnthropicGateway(api_key="test-key")

        with patch("langfuse.get_client", return_value=mock_langfuse):
            result = gateway.send_message_traced(
                "1 + 1 = ",
                name="calculator",
                system="You are a calculator. Output only the result.",
            )

        assert result == "2"
        call_kwargs = mock_anthropic_client.messages.create.call_args
        assert call_kwargs.kwargs["system"] == "You are a calculator. Output only the result."
