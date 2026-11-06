"""LLM client: retry classification, backoff and DeepSeek API handling.

All API calls are mocked — no network access.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest

from polyreader.llm import create_llm_client
from polyreader.llm.client import LLMClient
from polyreader.llm.errors import EmptyResponseError, backoff_delay, is_transient_error
from polyreader.llm.providers import (
    api_key_env_var,
    resolve_deepseek_model,
    resolve_provider,
)

DEEPSEEK_URL = "https://api.deepseek.com"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
_REQUEST = httpx.Request("POST", f"{DEEPSEEK_URL}/chat/completions")


def _status_error(status: int, headers: dict | None = None) -> openai.APIStatusError:
    response = httpx.Response(status, request=_REQUEST, headers=headers or {})
    cls = {
        400: openai.BadRequestError,
        401: openai.AuthenticationError,
        404: openai.NotFoundError,
        422: openai.UnprocessableEntityError,
        429: openai.RateLimitError,
    }.get(status, openai.InternalServerError if status >= 500 else openai.APIStatusError)
    return cls(f"Error code: {status}", response=response, body=None)


def _response(content="ok", finish_reason="stop", tool_calls=None, reasoning=None, usage=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=reasoning)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=usage,
    )


@pytest.fixture
def no_sleep(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr("polyreader.llm.client.asyncio.sleep", sleep)
    return sleep


def _client_with(create: AsyncMock, **kwargs) -> LLMClient:
    llm = LLMClient(api_key="k", base_url=kwargs.pop("base_url", DEEPSEEK_URL),
                    model=kwargs.pop("model", "deepseek-flash"), **kwargs)
    fake = MagicMock()
    fake.chat.completions.create = create
    llm._client = fake
    return llm


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


class TestIsTransientError:
    @pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504])
    def test_transient_status(self, status):
        assert is_transient_error(_status_error(status)) is True

    @pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422])
    def test_permanent_status(self, status):
        """Bad request, auth, insufficient balance (402), unknown model, invalid params."""
        assert is_transient_error(_status_error(status)) is False

    def test_timeout_and_connection_errors(self):
        assert is_transient_error(openai.APITimeoutError(request=_REQUEST)) is True
        assert is_transient_error(openai.APIConnectionError(request=_REQUEST)) is True
        assert is_transient_error(httpx.ReadTimeout("read timed out")) is True

    def test_empty_response_is_transient(self):
        assert is_transient_error(EmptyResponseError("no choices")) is True

    def test_timed_out_message_is_transient(self):
        """openai's timeout text "Request timed out." was missed by the old heuristic."""
        assert is_transient_error(Exception("Request timed out.")) is True

    def test_permanent_status_overrides_message(self):
        """A 400 whose message mentions 'connection' must still not be retried."""
        err = _status_error(400)
        err.message = "connection parameter invalid"
        assert is_transient_error(err) is False

    def test_wrapped_cause(self):
        wrapper = RuntimeError("instructor gave up")
        wrapper.__cause__ = _status_error(503)
        assert is_transient_error(wrapper) is True

    def test_backoff_bounded_and_retry_after(self):
        assert backoff_delay(0) == 1
        assert backoff_delay(2) == 4
        assert backoff_delay(20) == 60
        err = _status_error(429, headers={"retry-after": "7"})
        assert backoff_delay(0, err) == 7
        err = _status_error(429, headers={"retry-after": "3600"})
        assert backoff_delay(0, err) == 60


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


class TestRetries:
    @pytest.mark.asyncio
    async def test_chat_retries_transient_then_succeeds(self, no_sleep):
        create = AsyncMock(side_effect=[_status_error(503), _response("done")])
        llm = _client_with(create)
        assert await llm.chat([{"role": "user", "content": "hi"}]) == "done"
        assert create.await_count == 2

    @pytest.mark.asyncio
    async def test_chat_does_not_retry_permanent(self, no_sleep):
        create = AsyncMock(side_effect=_status_error(401))
        llm = _client_with(create)
        with pytest.raises(openai.AuthenticationError):
            await llm.chat([{"role": "user", "content": "hi"}])
        assert create.await_count == 1
        no_sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_insufficient_balance_not_retried(self, no_sleep):
        create = AsyncMock(side_effect=_status_error(402))
        llm = _client_with(create)
        with pytest.raises(openai.APIStatusError):
            await llm.chat([{"role": "user", "content": "hi"}])
        assert create.await_count == 1

    @pytest.mark.asyncio
    async def test_blank_content_retried(self, no_sleep):
        """DeepSeek JSON mode may return whitespace-only content."""
        create = AsyncMock(side_effect=[_response("   "), _response(None), _response("{}")])
        llm = _client_with(create)
        assert await llm.chat([{"role": "user", "content": "json"}]) == "{}"
        assert create.await_count == 3

    @pytest.mark.asyncio
    async def test_retries_exhausted_raises(self, no_sleep):
        create = AsyncMock(side_effect=_status_error(503))
        llm = _client_with(create)
        with pytest.raises(openai.InternalServerError):
            await llm.chat([{"role": "user", "content": "hi"}], max_retries=3)
        assert create.await_count == 3

    @pytest.mark.asyncio
    async def test_insufficient_system_resource_retried(self, no_sleep):
        create = AsyncMock(side_effect=[
            _response("partial", finish_reason="insufficient_system_resource"),
            _response("full"),
        ])
        llm = _client_with(create)
        assert await llm.chat([{"role": "user", "content": "hi"}]) == "full"

    @pytest.mark.asyncio
    async def test_tool_call_transient_retry(self, no_sleep):
        """chat_with_tools previously had no retry at all."""
        create = AsyncMock(side_effect=[openai.APIConnectionError(request=_REQUEST), _response("answer")])
        llm = _client_with(create)
        result = await llm.chat_with_tools([{"role": "user", "content": "x"}], tools=[])
        assert result.content == "answer"
        assert create.await_count == 2

    @pytest.mark.asyncio
    async def test_no_choices_retried(self, no_sleep):
        create = AsyncMock(side_effect=[SimpleNamespace(choices=[], usage=None), _response("ok")])
        llm = _client_with(create)
        assert await llm.chat([{"role": "user", "content": "hi"}]) == "ok"


# ---------------------------------------------------------------------------
# DeepSeek API usage
# ---------------------------------------------------------------------------


class TestDeepSeek:
    def test_legacy_model_names(self):
        assert resolve_deepseek_model("deepseek-reasoner") == ("deepseek-flash", True)
        assert resolve_deepseek_model("deepseek-chat") == ("deepseek-flash", False)
        assert resolve_deepseek_model("deepseek/deepseek-reasoner") == ("deepseek-flash", True)
        assert resolve_deepseek_model("deepseek-v4-flash") == ("deepseek-flash", None)
        assert resolve_deepseek_model("deepseek-v4-pro") == ("deepseek-v4-pro", None)
        assert resolve_deepseek_model("") == ("deepseek-flash", None)

    def test_defaults_thinking_on_and_long_timeout(self):
        llm = LLMClient(api_key="k", base_url=DEEPSEEK_URL, model="deepseek-flash")
        assert llm._uses_thinking_mode() is True
        assert llm.timeout == 600
        assert llm._build_extra_body() == {"thinking": {"type": "enabled"}}

    def test_legacy_chat_runs_without_thinking(self):
        llm = LLMClient(api_key="k", base_url=DEEPSEEK_URL, model="deepseek-chat")
        assert llm.model == "deepseek-flash"
        assert llm._uses_thinking_mode() is False
        assert llm._build_extra_body() == {"thinking": {"type": "disabled"}}

    def test_thinking_disabled_flag(self):
        llm = LLMClient(api_key="k", base_url=DEEPSEEK_URL, model="deepseek-flash",
                        thinking_disabled=True)
        assert llm._build_extra_body() == {"thinking": {"type": "disabled"}}

    @pytest.mark.asyncio
    async def test_request_shape_thinking(self, no_sleep):
        create = AsyncMock(return_value=_response("ok"))
        llm = _client_with(create, reasoning_effort="max")
        await llm.chat([{"role": "user", "content": "hi"}])
        kwargs = create.await_args.kwargs
        assert kwargs["model"] == "deepseek-flash"
        assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
        assert kwargs["reasoning_effort"] == "max"
        assert "temperature" not in kwargs  # no effect in thinking mode

    @pytest.mark.asyncio
    async def test_request_shape_non_thinking(self, no_sleep):
        create = AsyncMock(return_value=_response("ok"))
        llm = _client_with(create, model="deepseek-chat", reasoning_effort="high", temperature=0.3)
        await llm.chat([{"role": "user", "content": "hi"}])
        kwargs = create.await_args.kwargs
        assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
        assert kwargs["temperature"] == 0.3
        assert "reasoning_effort" not in kwargs

    def test_invalid_reasoning_effort(self):
        with pytest.raises(ValueError):
            LLMClient(api_key="k", base_url=DEEPSEEK_URL, model="deepseek-flash",
                      reasoning_effort="extreme")

    def test_structured_output_mode(self):
        """Thinking mode rejects forced tool_choice -> JSON mode; TOOLS otherwise."""
        import instructor

        thinking = LLMClient(api_key="k", base_url=DEEPSEEK_URL, model="deepseek-flash")
        assert thinking._get_or_create_instructor_client().mode == instructor.Mode.JSON
        plain = LLMClient(api_key="k", base_url=DEEPSEEK_URL, model="deepseek-chat")
        assert plain._get_or_create_instructor_client().mode == instructor.Mode.TOOLS

    @pytest.mark.asyncio
    async def test_tool_turn_keeps_reasoning_content(self, no_sleep):
        """With tools, DeepSeek requires reasoning_content of earlier turns (else 400)."""
        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "get_page_content"
        tool_call.function.arguments = '{"pages": "1"}'
        tool_call.model_dump.return_value = {
            "id": "call_1", "type": "function",
            "function": {"name": "get_page_content", "arguments": '{"pages": "1"}'},
        }
        create = AsyncMock(return_value=_response("", tool_calls=[tool_call], reasoning="think"))
        llm = _client_with(create)
        result = await llm.chat_with_tools([{"role": "user", "content": "x"}], tools=[{"type": "function"}])
        assert result.raw_message_dict["reasoning_content"] == "think"
        assert result.raw_message_dict["tool_calls"][0]["id"] == "call_1"

    @pytest.mark.asyncio
    async def test_usage_accumulates(self, no_sleep):
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, prompt_cache_hit_tokens=4,
                                completion_tokens_details=SimpleNamespace(reasoning_tokens=3))
        create = AsyncMock(return_value=_response("ok", usage=usage))
        llm = _client_with(create)
        await llm.chat([{"role": "user", "content": "a"}])
        await llm.chat([{"role": "user", "content": "b"}])
        assert llm.usage["requests"] == 2
        assert llm.usage["prompt_tokens"] == 20
        assert llm.usage["reasoning_tokens"] == 6
        assert llm.usage["prompt_cache_hit_tokens"] == 8

    def test_openrouter_behaviour_unchanged(self):
        llm = create_llm_client(api_key="k", model="deepseek/deepseek-reasoner", base_url=OPENROUTER_URL)
        assert llm.model == "deepseek/deepseek-reasoner"
        assert llm.timeout == 120
        assert llm._build_extra_body() is None
        assert llm._uses_thinking_mode() is True  # reasoner -> MD_JSON as before


class TestProviderResolution:
    def test_api_key_env_var(self):
        assert api_key_env_var(DEEPSEEK_URL) == "DEEPSEEK_API_KEY"
        assert api_key_env_var(OPENROUTER_URL) == "OPENROUTER_API_KEY"
        assert api_key_env_var("https://my-proxy.example/v1") == "OPENROUTER_API_KEY"

    def test_auto_prefers_openrouter_then_deepseek(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
        assert resolve_provider("auto", None) == ("deepseek", DEEPSEEK_URL)
        monkeypatch.setenv("OPENROUTER_API_KEY", "y")
        assert resolve_provider("auto", None) == ("openrouter", OPENROUTER_URL)

    def test_explicit_base_url_wins(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "y")
        assert resolve_provider("openrouter", DEEPSEEK_URL) == ("deepseek", DEEPSEEK_URL)

    def test_extraction_config_uses_deepseek_key(self, monkeypatch):
        from polyreader.extraction import ExtractionConfig

        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        assert ExtractionConfig(base_url=DEEPSEEK_URL).api_key == "ds-key"
        assert ExtractionConfig().api_key == "or-key"

    def test_indexing_config_uses_deepseek_key(self, monkeypatch):
        from polyreader.indexing.config import ConfigLoader

        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        assert ConfigLoader().load({"base_url": DEEPSEEK_URL}).api_key == "ds-key"
        assert ConfigLoader().load({}).api_key == "or-key"

    def test_vision_only_for_flash(self):
        from polyreader.extraction import ExtractionConfig
        from polyreader.extraction.tools import supports_vision

        assert supports_vision(ExtractionConfig(base_url=DEEPSEEK_URL, model="deepseek-flash",
                                                 vision_enabled=True, api_key="k")) is True
        assert supports_vision(ExtractionConfig(base_url=DEEPSEEK_URL, model="deepseek-v4-pro",
                                                 vision_enabled=True, api_key="k")) is False
