"""Unified OpenAI-compatible async LLM client (OpenRouter, DeepSeek API, ...)."""

import asyncio
import base64
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import openai
from pydantic import BaseModel

from .errors import EmptyResponseError, backoff_delay, is_transient_error
from .providers import (
    DEEPSEEK,
    OPENROUTER,
    REASONING_EFFORTS,
    default_timeout,
    detect_provider,
    resolve_deepseek_model,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# finish_reason values that mean the provider gave up mid-generation (DeepSeek).
_TRANSIENT_FINISH_REASONS = frozenset({"insufficient_system_resource", "aborted"})


@dataclass
class ToolCallResult:
    """A single tool call from the LLM response."""
    id: str
    name: str
    arguments: str  # JSON string


@dataclass
class ChatWithToolsResponse:
    """Response from chat_with_tools(), provider-agnostic."""
    content: str
    tool_calls: list[ToolCallResult]
    raw_message_dict: dict  # Ready to append to messages list
    reasoning_content: str | None = None


def strip_tool_messages(messages: list[dict]) -> list[dict]:
    """Convert tool-calling conversation history into plain messages.

    DeepSeek's API rejects tool-protocol messages (role: "tool", tool_calls arrays)
    when the request doesn't include a `tools` parameter. This helper flattens
    such messages into plain assistant/user text so the conversation can be reused
    in non-tool API calls (wrap-up, Instructor MD_JSON, etc.).

    - Assistant messages with tool_calls → text: [Called tool: name(args)]
    - Tool role messages → user role: [Tool result for name]:\n{content}
    - reasoning_content preserved on assistant messages
    - System / plain user / assistant messages pass through unchanged
    """
    # Build lookup: call_id -> function name
    call_id_to_name: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                call_id_to_name[tc["id"]] = tc["function"]["name"]

    cleaned: list[dict] = []
    for msg in messages:
        role = msg.get("role", "")

        if role == "assistant" and msg.get("tool_calls"):
            # Flatten tool_calls into readable text
            lines = []
            if msg.get("content"):
                lines.append(msg["content"])
            for tc in msg["tool_calls"]:
                name = tc["function"]["name"]
                args = tc["function"].get("arguments", "")
                lines.append(f"[Called tool: {name}({args})]")
            new_msg: dict = {"role": "assistant", "content": "\n".join(lines)}
            if "reasoning_content" in msg:
                new_msg["reasoning_content"] = msg["reasoning_content"]
            cleaned.append(new_msg)

        elif role == "tool":
            # Convert to user message referencing the tool name
            call_id = msg.get("tool_call_id", "")
            tool_name = call_id_to_name.get(call_id, "unknown")
            content = msg.get("content", "")
            cleaned.append({
                "role": "user",
                "content": f"[Tool result for {tool_name}]:\n{content}",
            })

        else:
            # System, user, plain assistant — pass through
            cleaned.append(msg)

    return cleaned



def clear_reasoning_content(messages: list[dict]) -> None:
    """Null out reasoning_content on all assistant messages in-place.

    Only safe for requests WITHOUT ``tools``: the DeepSeek API requires the
    reasoning_content of every previous assistant turn to be sent back when a
    request carries ``tools`` and rejects the request (HTTP 400) otherwise.
    """
    for msg in messages:
        if msg.get("role") == "assistant" and "reasoning_content" in msg:
            msg["reasoning_content"] = None


class LLMClient:
    """OpenAI-compatible async client supporting text, vision and tools.

    Retries transient failures (connection errors, timeouts, 429, 5xx, empty
    responses) with bounded exponential backoff and raises permanent ones
    (400/401/402/403/404/422) immediately.
    """

    def _is_reasoner_model(self) -> bool:
        """Intrinsic model property: always-on reasoning, rejects tool messages.

        Distinct from thinking_enabled, which is a user opt-in for extra_body.
        These are orthogonal: a reasoner model always needs MD_JSON and tool
        stripping regardless of whether thinking_enabled is set.
        """
        return "reasoner" in self.model.lower()

    def _uses_thinking_mode(self) -> bool:
        """True when requests run in thinking (reasoning) mode.

        Thinking mode rejects forced ``tool_choice``, so structured output
        cannot use Instructor's TOOLS mode.
        """
        if self.provider == DEEPSEEK:
            return self._deepseek_thinking
        return self._is_reasoner_model()

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int | None = None,
        temperature: float = 1.0,
        max_concurrent: int = 1,
        thinking_enabled: bool = False,
        include_extra_body: bool = True,
        default_headers: dict[str, str] | None = None,
        reasoning_effort: str | None = None,
        thinking_disabled: bool = False,
    ):
        self.provider = detect_provider(base_url)
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout if timeout else default_timeout(base_url)
        self.temperature = temperature
        self.thinking_enabled = thinking_enabled
        self.include_extra_body = include_extra_body
        self.default_headers = default_headers or {}
        if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(
                f"reasoning_effort must be one of {REASONING_EFFORTS}, got '{reasoning_effort}'"
            )
        self.reasoning_effort = reasoning_effort
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._client: openai.AsyncOpenAI | None = None
        self._instructor_client = None
        # Accumulated token usage across all calls made by this client.
        self.usage: dict[str, int] = {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "prompt_cache_hit_tokens": 0,
        }

        if self.provider == DEEPSEEK:
            # Current DeepSeek models think by default; retired names imply a mode.
            self.model, implied = resolve_deepseek_model(model)
            if self.model != model:
                logger.info(f"DeepSeek model '{model}' -> '{self.model}'")
            if thinking_disabled:
                self._deepseek_thinking = False
            elif thinking_enabled:
                self._deepseek_thinking = True
            else:
                self._deepseek_thinking = implied if implied is not None else True
            logger.info(
                f"DeepSeek API: model={self.model}, "
                f"thinking={'enabled' if self._deepseek_thinking else 'disabled'}"
                + (f", reasoning_effort={reasoning_effort}" if reasoning_effort else "")
            )
        else:
            self.model = model
            self._deepseek_thinking = False
            if self.thinking_enabled:
                logger.info(f"Thinking mode enabled for model '{self.model}'")
            if "/" not in self.model and self.provider == OPENROUTER:
                logger.warning(
                    f"Model '{self.model}' has no provider prefix. "
                    f"OpenRouter expects 'provider/model-name' format "
                    f"(e.g. 'deepseek/deepseek-reasoner')."
                )

    def _get_or_create_client(self) -> openai.AsyncOpenAI:
        """Lazy singleton — reuses TCP/SSL connections across calls."""
        if self._client is None:
            headers = self.default_headers
            if not headers and self.provider == OPENROUTER:
                headers = {
                    "HTTP-Referer": "https://github.com/Chenghao-Wu/polyextract",
                    "X-Title": "polyreader",
                }
            self._client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=0,  # disable SDK-level retries; LLMClient manages its own
                default_headers=headers or None,
            )
        return self._client

    def _get_or_create_instructor_client(self):
        """Create Instructor-wrapped client for structured outputs.

        Thinking mode rejects forced tool calls, so it needs a JSON mode: the
        DeepSeek API's native JSON Output (response_format=json_object) where
        available, Markdown-fenced JSON otherwise.
        """
        if self._instructor_client is None:
            import instructor
            client = self._get_or_create_client()
            if not self._uses_thinking_mode():
                mode = instructor.Mode.TOOLS
            elif self.provider == DEEPSEEK:
                mode = instructor.Mode.JSON
            else:
                mode = instructor.Mode.MD_JSON
            self._instructor_client = instructor.from_openai(client, mode=mode)
        return self._instructor_client

    def _build_extra_body(self) -> dict | None:
        """Build extra_body for thinking control.

        DeepSeek: always state the mode explicitly (the API defaults to
        thinking). Other providers: send it only when thinking_enabled.
        """
        if not self.include_extra_body:
            return None
        if self.provider == DEEPSEEK:
            return {"thinking": {"type": "enabled" if self._deepseek_thinking else "disabled"}}
        if self.thinking_enabled:
            return {"thinking": {"type": "enabled"}}
        return None

    def _request_kwargs(self, messages: list[dict], temperature: float, **extra) -> dict:
        kwargs = dict(model=self.model, messages=messages, **extra)
        thinking = self._uses_thinking_mode()
        # DeepSeek ignores temperature in thinking mode; don't send it there.
        if not (self.provider == DEEPSEEK and thinking):
            kwargs["temperature"] = temperature
        if self.reasoning_effort and thinking and self.provider == DEEPSEEK:
            kwargs["reasoning_effort"] = self.reasoning_effort
        extra_body = self._build_extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    def _record_usage(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.usage["requests"] += 1
        self.usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        self.usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        details = getattr(usage, "completion_tokens_details", None)
        self.usage["reasoning_tokens"] += getattr(details, "reasoning_tokens", 0) or 0
        self.usage["prompt_cache_hit_tokens"] += getattr(usage, "prompt_cache_hit_tokens", 0) or 0

    async def _create_completion(
        self,
        kwargs: dict,
        max_retries: int,
        label: str,
        require_content: bool = False,
    ):
        """Call chat.completions.create, retrying only transient failures.

        Transient = connection errors, timeouts, 408/409/425/429, 5xx, empty
        or aborted responses (and, with ``require_content``, blank message
        content). Backoff is exponential (1s, 2s, 4s, ... capped at 60s) and
        honours Retry-After. Permanent errors (400 bad request, 401/403 auth,
        402 insufficient balance, 404, 422 invalid parameters) raise at once.
        """
        max_retries = max(1, max_retries)
        for attempt in range(max_retries):
            try:
                client = self._get_or_create_client()
                response = await client.chat.completions.create(**kwargs)
                self._record_usage(response)
                if not getattr(response, "choices", None):
                    raise EmptyResponseError(f"{label}: provider returned no choices")
                choice = response.choices[0]
                if choice.finish_reason in _TRANSIENT_FINISH_REASONS:
                    raise EmptyResponseError(
                        f"{label}: generation stopped early (finish_reason={choice.finish_reason})"
                    )
                if require_content and not (choice.message.content or "").strip():
                    raise EmptyResponseError(f"{label}: response has no message content")
                return response
            except Exception as e:
                if not is_transient_error(e):
                    logger.error(f"{label} failed with a permanent error: {type(e).__name__}: {e}")
                    raise
                if attempt == max_retries - 1:
                    logger.error(f"{label}: max retries reached ({max_retries}): {type(e).__name__}: {e}")
                    raise
                wait = backoff_delay(attempt, e)
                logger.warning(
                    f"{label} failed (attempt {attempt + 1}/{max_retries}): "
                    f"{type(e).__name__}; retrying in {wait:.0f}s"
                )
                await asyncio.sleep(wait)

    async def chat(
        self,
        messages: list[dict],
        temperature: float = None,
        max_retries: int = 3,
        json_mode: bool = False,
    ) -> str:
        """Text completion with retries and exponential backoff.

        ``json_mode=True`` requests the DeepSeek API's JSON Output
        (``response_format=json_object``; the prompt must ask for JSON).
        Other providers ignore it and rely on the prompt.
        """
        temperature = self.temperature if temperature is None else temperature
        extra = {"response_format": {"type": "json_object"}} if json_mode and self.provider == DEEPSEEK else {}
        kwargs = self._request_kwargs(messages, temperature, **extra)
        async with self._semaphore:
            response = await self._create_completion(
                kwargs, max_retries, "LLM call", require_content=True,
            )
        return response.choices[0].message.content.strip()

    async def chat_simple(
        self, prompt: str, temperature: float = None
    ) -> str:
        """Single-prompt convenience method."""
        return await self.chat(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
        )

    async def chat_json(
        self, messages: list[dict], temperature: float = None
    ) -> dict:
        """Chat with JSON output parsing."""
        from polyreader.utils.json_utils import extract_json

        response = await self.chat(messages, temperature=temperature)
        return extract_json(response)

    async def chat_structured(
        self,
        response_model: type[T],
        messages: list[dict],
        max_retries: int = 3,
        temperature: float = None,
        transport_retries: int = 3,
        **kwargs
    ) -> T:
        """Chat with structured output using Instructor.

        Args:
            response_model: Pydantic model class for structured response.
            messages: Conversation messages.
            max_retries: Maximum retries on validation errors (Instructor auto-retries).
            temperature: Sampling temperature.
            transport_retries: Attempts for transient API failures.
            **kwargs: Additional arguments passed to the API.

        Returns:
            Validated instance of response_model.
        """
        temperature = self.temperature if temperature is None else temperature
        merged_kwargs = {k: v for k, v in kwargs.items() if k not in ("temperature", "max_retries")}
        request = self._request_kwargs(messages, temperature, **merged_kwargs)
        transport_retries = max(1, transport_retries)
        async with self._semaphore:
            client = self._get_or_create_instructor_client()
            for attempt in range(transport_retries):
                try:
                    return await client.chat.completions.create(
                        response_model=response_model,
                        max_retries=max_retries,
                        **request,
                    )
                except Exception as e:
                    if not is_transient_error(e) or attempt == transport_retries - 1:
                        raise
                    wait = backoff_delay(attempt, e)
                    logger.warning(
                        f"Structured output call failed (attempt {attempt + 1}/{transport_retries}): "
                        f"{type(e).__name__}; retrying in {wait:.0f}s"
                    )
                    await asyncio.sleep(wait)

    async def chat_vision(
        self,
        prompt: str,
        images: list[str],
        temperature: float = None,
        max_retries: int = 5,
    ) -> str:
        """Vision completion with base64 images.

        Args:
            prompt: Text prompt
            images: List of base64-encoded image strings or file paths
        """
        temperature = self.temperature if temperature is None else temperature
        content = [{"type": "text", "text": prompt}]
        for img in images:
            # Short strings might be file paths; long strings are base64 data
            is_file = len(img) < 1024 and Path(img).exists()
            if is_file:
                with open(img, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode()
                ext = Path(img).suffix.lstrip(".")
                if ext == "jpg":
                    ext = "jpeg"
                media_type = f"image/{ext}"
            else:
                img_data = img
                media_type = "image/png"

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{img_data}"
                },
            })

        messages = [{"role": "user", "content": content}]
        kwargs = self._request_kwargs(messages, temperature)
        async with self._semaphore:
            response = await self._create_completion(
                kwargs, max_retries, "Vision call", require_content=True,
            )
        return response.choices[0].message.content.strip()

    async def chat_with_finish_reason(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
        max_retries: int = 1,
    ) -> tuple[str, str]:
        """Chat returning (content, finish_reason).

        No retry loop by default (``max_retries=1``); pass a larger value to
        retry transient failures.
        """
        temperature = self.temperature if temperature is None else temperature
        extra = {"max_tokens": max_tokens} if max_tokens is not None else {}
        kwargs = self._request_kwargs(messages, temperature, **extra)
        async with self._semaphore:
            response = await self._create_completion(kwargs, max_retries, "LLM call")
        content = response.choices[0].message.content or ""
        finish = response.choices[0].finish_reason
        reason = "finished" if finish != "length" else "max_output_reached"
        return content.strip(), reason

    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_choice: str = "auto",
        temperature: float = None,
        max_retries: int = 3,
    ) -> ChatWithToolsResponse:
        """Chat with function-calling tools. Returns provider-agnostic response.

        Args:
            messages: Conversation history. Assistant messages must keep their
                ``reasoning_content`` (``raw_message_dict`` does) — DeepSeek
                thinking mode rejects tool requests without it.
            tools: OpenAI-format tool definitions.
            tool_choice: "auto" or "none" ("required" is rejected by thinking models).
            temperature: Sampling temperature.
            max_retries: Attempts for transient API failures.

        Returns:
            ChatWithToolsResponse with content, tool_calls, and raw_message_dict.
        """
        temperature = self.temperature if temperature is None else temperature

        # For reasoner models doing wrap-up, strip tool messages and skip tools param
        if self._is_reasoner_model() and tool_choice == "none":
            kwargs = self._request_kwargs(strip_tool_messages(messages), temperature)
        else:
            kwargs = self._request_kwargs(messages, temperature, tools=tools, tool_choice=tool_choice)

        response = await self._create_completion(kwargs, max_retries, "Tool call")
        msg = response.choices[0].message

        # Build raw_message_dict for appending to conversation
        msg_dict: dict = {"role": "assistant", "content": msg.content or ""}
        reasoning = getattr(msg, "reasoning_content", None) or None
        if reasoning:
            msg_dict["reasoning_content"] = reasoning
        elif self.thinking_enabled or self._uses_thinking_mode():
            msg_dict["reasoning_content"] = None

        tool_call_results = []
        if msg.tool_calls:
            msg_dict["tool_calls"] = [
                tc.model_dump(exclude_none=True) for tc in msg.tool_calls
            ]
            tool_call_results = [
                ToolCallResult(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                )
                for tc in msg.tool_calls
            ]

        return ChatWithToolsResponse(
            content=msg.content or "",
            tool_calls=tool_call_results,
            raw_message_dict=msg_dict,
            reasoning_content=reasoning,
        )
