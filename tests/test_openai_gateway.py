from unittest.mock import MagicMock, patch

import pytest

from gateways.openai_gateway import OpenAiGateway


@pytest.fixture
def mock_openai_client():
    with patch("gateways.openai_gateway.OpenAI") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_instance


class TestOpenAiGatewayInit:
    def test_uses_provided_api_key(self, mock_openai_client):
        OpenAiGateway(api_key="test-key-123")

        from gateways.openai_gateway import OpenAI
        OpenAI.assert_called_once_with(api_key="test-key-123")

    def test_falls_back_to_env_var_when_no_key_provided(self, mock_openai_client):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key-456"}, clear=False):
            OpenAiGateway()

        from gateways.openai_gateway import OpenAI
        OpenAI.assert_called_once_with(api_key="env-key-456")


class TestOpenAiGatewaySendMessage:
    def test_send_message_returns_output_text(self, mock_openai_client):
        mock_response = MagicMock()
        mock_response.output_text = "Once upon a time..."
        mock_openai_client.responses.create.return_value = mock_response

        gateway = OpenAiGateway(api_key="test-key")
        result = gateway.send_message("Write a bedtime story")

        assert result == "Once upon a time..."
        mock_openai_client.responses.create.assert_called_once_with(
            model="gpt-5.5",
            input="Write a bedtime story",
        )

    def test_send_message_uses_custom_model(self, mock_openai_client):
        mock_response = MagicMock()
        mock_response.output_text = "Hi"
        mock_openai_client.responses.create.return_value = mock_response

        gateway = OpenAiGateway(api_key="test-key")
        gateway.send_message("Hello", model="gpt-4o")

        call_kwargs = mock_openai_client.responses.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o"


class TestOpenAiGatewayStreamMessage:
    def test_stream_message_yields_text_deltas(self, mock_openai_client):
        delta_event_1 = MagicMock(type="response.output_text.delta", delta="Once ")
        delta_event_2 = MagicMock(type="response.output_text.delta", delta="upon ")
        delta_event_3 = MagicMock(type="response.done")
        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter([delta_event_1, delta_event_2, delta_event_3]))
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_openai_client.responses.create.return_value = mock_stream

        gateway = OpenAiGateway(api_key="test-key")
        result = list(gateway.stream_message("Write a story"))

        assert result == ["Once ", "upon "]

    def test_stream_message_calls_with_stream_true(self, mock_openai_client):
        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter([]))
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_openai_client.responses.create.return_value = mock_stream

        gateway = OpenAiGateway(api_key="test-key")
        list(gateway.stream_message("Hello"))

        call_kwargs = mock_openai_client.responses.create.call_args
        assert call_kwargs.kwargs["stream"] is True
        assert call_kwargs.kwargs["model"] == "gpt-5.5"


class TestOpenAiGatewaySendMessageTraced:
    def test_send_message_traced_returns_completion_text(self, mock_openai_client):
        mock_completion = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "2"
        mock_completion.choices = [MagicMock(message=mock_message)]

        mock_langfuse_openai = MagicMock()
        mock_langfuse_openai.chat.completions.create.return_value = mock_completion

        gateway = OpenAiGateway(api_key="test-key")

        with patch("langfuse.openai.openai", mock_langfuse_openai):
            result = gateway.send_message_traced(
                "1 + 1 = ",
                name="test-chat",
                metadata={"someMetadataKey": "someValue"},
            )

        assert result == "2"
        mock_langfuse_openai.chat.completions.create.assert_called_once_with(
            name="test-chat",
            model="gpt-4o",
            messages=[{"role": "user", "content": "1 + 1 = "}],
            metadata={"someMetadataKey": "someValue"},
        )

    def test_send_message_traced_with_system_prompt(self, mock_openai_client):
        mock_completion = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "2"
        mock_completion.choices = [MagicMock(message=mock_message)]

        mock_langfuse_openai = MagicMock()
        mock_langfuse_openai.chat.completions.create.return_value = mock_completion

        gateway = OpenAiGateway(api_key="test-key")

        with patch("langfuse.openai.openai", mock_langfuse_openai):
            result = gateway.send_message_traced(
                "1 + 1 = ",
                name="calculator",
                system="You are a calculator. Output only the result.",
            )

        assert result == "2"
        call_kwargs = mock_langfuse_openai.chat.completions.create.call_args
        assert call_kwargs.kwargs["messages"][0] == {
            "role": "system",
            "content": "You are a calculator. Output only the result.",
        }
        assert call_kwargs.kwargs["messages"][1] == {
            "role": "user",
            "content": "1 + 1 = ",
        }

    def test_send_message_traced_uses_custom_model(self, mock_openai_client):
        mock_completion = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Hi"
        mock_completion.choices = [MagicMock(message=mock_message)]

        mock_langfuse_openai = MagicMock()
        mock_langfuse_openai.chat.completions.create.return_value = mock_completion

        gateway = OpenAiGateway(api_key="test-key")

        with patch("langfuse.openai.openai", mock_langfuse_openai):
            gateway.send_message_traced("Hello", name="test", model="gpt-5.5")

        call_kwargs = mock_langfuse_openai.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-5.5"
