import os

from anthropic import Anthropic

from gateways.base import BaseGateway


class AnthropicGateway(BaseGateway):
    """Gateway for interacting with the Anthropic API."""

    def __init__(self, api_key: str | None = None):
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def send_message(self, message: str, model: str = "claude-opus-4-6", max_tokens: int = 1024) -> str:
        """Send a message to the Anthropic API and return the response."""
        response = self.client.messages.create(
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": message}],
            model=model,
        )
        return response.content[0].text

    def stream_message(self, message: str, model: str = "claude-opus-4-6", max_tokens: int = 1024):
        """Stream a message from the Anthropic API, yielding content deltas."""
        with self.client.messages.stream(
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": message}],
            model=model,
        ) as stream:
            for text in stream.text_stream:
                yield text

    def send_message_traced(
        self,
        message: str,
        *,
        name: str,
        model: str = "claude-opus-4-6",
        max_tokens: int = 1024,
        system: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Send a message with Langfuse tracing via OpenTelemetry.

        Requires Langfuse + OTEL instrumentation to be initialized before use:
            from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
            AnthropicInstrumentor().instrument()
        """
        from langfuse import get_client

        langfuse = get_client()
        messages = [{"role": "user", "content": message}]

        with langfuse.start_as_current_observation(
            as_type="generation",
            name=name,
            metadata=metadata,
            model=model,
            input=messages,
        ):
            kwargs: dict = {
                "max_tokens": max_tokens,
                "messages": messages,
                "model": model,
            }
            if system:
                kwargs["system"] = system

            response = self.client.messages.create(**kwargs)
            result = response.content[0].text

            langfuse.update_current_generation(output=result)
            return result
