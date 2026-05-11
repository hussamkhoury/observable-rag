import os

from dotenv import load_dotenv

from gateways.anthropic_gateway import AnthropicGateway
from gateways.openai_gateway import OpenAiGateway

load_dotenv()


def setup_langfuse():
    """Initialize Langfuse SDK pointing to local Docker instance."""
    os.environ.setdefault("LANGFUSE_HOST", "http://localhost:3000")
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", os.environ.get("LANGFUSE_PUBLIC_KEY", ""))
    os.environ.setdefault("LANGFUSE_SECRET_KEY", os.environ.get("LANGFUSE_SECRET_KEY", ""))


def setup_otel_anthropic():
    """Instrument Anthropic SDK with OpenTelemetry for Langfuse tracing."""
    from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

    AnthropicInstrumentor().instrument()


def main():
    setup_langfuse()
    setup_otel_anthropic()

    anthropic = AnthropicGateway()
    openai = OpenAiGateway()

    message = "1 + 1 = "
    system_prompt = "You are a very accurate calculator. You output only the result of the calculation."

    print("=== Anthropic (traced) ===")
    result_claude = anthropic.send_message_traced(
        message,
        name="anthropic-calculator",
        system=system_prompt,
        metadata={"provider": "anthropic", "mode": "traced"},
    )
    print(f"Claude: {result_claude}")

    print("\n=== OpenAI (traced) ===")
    result_gpt = openai.send_message_traced(
        message,
        name="openai-calculator",
        system=system_prompt,
        metadata={"provider": "openai", "mode": "traced"},
    )
    print(f"GPT: {result_gpt}")

    print("\n=== Anthropic (standard) ===")
    result_std = anthropic.send_message(message)
    print(f"Claude (std): {result_std}")

    print("\n=== OpenAI (standard) ===")
    result_std_oai = openai.send_message(message)
    print(f"GPT (std): {result_std_oai}")

    print("\nDone. Check http://localhost:3000 for traces.")


if __name__ == "__main__":
    main()
