import os

from openai import OpenAI

from gateways.base import BaseGateway


class OpenAiGateway(BaseGateway):
    """Gateway for interacting with the OpenAI API."""

    def __init__(self, api_key: str | None = None):
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def send_message(self, message: str, model: str = "gpt-5.5") -> str:
        """Send a message to the OpenAI API and return the response."""
        response = self.client.responses.create(
            model=model,
            input=message,
        )
        return response.output_text

    def stream_message(self, message: str, model: str = "gpt-5.5"):
        """Stream a message from the OpenAI API, yielding content deltas."""
        with self.client.responses.create(
            model=model,
            input=message,
            stream=True,
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    yield event.delta

    def send_message_traced(
        self,
        message: str,
        *,
        name: str,
        model: str = "gpt-4o",
        system: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Send a message with Langfuse tracing via chat.completions.

        Requires Langfuse to be initialized before use:
            from langfuse.openai import openai  # replaces the client
        """
        from langfuse.openai import openai as langfuse_openai

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        completion = langfuse_openai.chat.completions.create(
            name=name,
            model=model,
            messages=messages,
            metadata=metadata,
        )
        return completion.choices[0].message.content
