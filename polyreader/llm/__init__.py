from .client import LLMClient, ToolCallResult, ChatWithToolsResponse, clear_reasoning_content
from .token_counter import count_tokens


def create_llm_client(
    api_key: str,
    model: str,
    base_url: str = "https://openrouter.ai/api/v1",
    temperature: float = 1.0,
    timeout: int | None = None,
    max_concurrent: int = 1,
    thinking_enabled: bool = False,
    include_extra_body: bool = True,
    reasoning_effort: str | None = None,
    thinking_disabled: bool = False,
):
    """Create an OpenAI-compatible LLM client (OpenRouter, DeepSeek API, ...).

    ``timeout=None`` uses the provider default (600 s for the DeepSeek API,
    120 s otherwise). ``reasoning_effort`` / ``thinking_disabled`` apply to
    the DeepSeek API.
    """
    return LLMClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        timeout=timeout,
        max_concurrent=max_concurrent,
        thinking_enabled=thinking_enabled,
        include_extra_body=include_extra_body,
        reasoning_effort=reasoning_effort,
        thinking_disabled=thinking_disabled,
    )
