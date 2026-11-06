"""LLM provider settings: OpenRouter (default) and the DeepSeek API.

Both are OpenAI-compatible; they differ in base URL, API-key variable, model
naming (OpenRouter uses ``provider/model`` slugs, DeepSeek bare names) and how
reasoning ("thinking") is controlled.

DeepSeek API facts this module encodes (api-docs.deepseek.com, Nov 2026):

- Models: ``deepseek-flash`` (DeepSeek-V4.1-Flash, text + image input, 1M
  context) and ``deepseek-v4-pro`` (being phased out; currently served by
  V4.1-Flash). ``deepseek-chat`` / ``deepseek-reasoner`` are retired legacy
  names (non-thinking / thinking), ``deepseek-v4-flash`` and
  ``deepseek-v4-flash-vision-exp`` are retired names routed to Flash.
- Thinking mode is ON by default. It is toggled with
  ``extra_body={"thinking": {"type": "enabled" | "disabled"}}`` and its depth
  set with ``reasoning_effort`` = ``low`` | ``high`` | ``max``.
- Thinking mode ignores ``temperature`` and rejects ``tool_choice`` values
  ``"required"`` or a named function (HTTP 400).
- With ``tools`` in a request, the ``reasoning_content`` of all previous
  assistant turns must be sent back, otherwise the API returns HTTP 400.
- A request that has not started inference after 10 minutes is closed.
"""

from __future__ import annotations

import os

OPENROUTER = "openrouter"
DEEPSEEK = "deepseek"
OTHER = "other"

PROVIDER_BASE_URLS = {
    OPENROUTER: "https://openrouter.ai/api/v1",
    DEEPSEEK: "https://api.deepseek.com",
}

PROVIDER_API_KEY_ENV = {
    OPENROUTER: "OPENROUTER_API_KEY",
    DEEPSEEK: "DEEPSEEK_API_KEY",
}

# Default extraction model per provider.
DEFAULT_MODELS = {
    OPENROUTER: "deepseek/deepseek-reasoner",
    DEEPSEEK: "deepseek-flash",
}

# Per-request timeout (seconds). DeepSeek thinking requests can run for
# minutes and the server itself gives up after 10 minutes without inference.
DEFAULT_TIMEOUTS = {
    DEEPSEEK: 600,
}
FALLBACK_TIMEOUT = 120

DEEPSEEK_MODELS = ("deepseek-flash", "deepseek-v4-pro")
# Only deepseek-flash accepts image input.
DEEPSEEK_VISION_MODELS = ("deepseek-flash",)

# Retired DeepSeek names -> (current model, thinking override or None).
DEEPSEEK_LEGACY_MODELS = {
    "deepseek-chat": ("deepseek-flash", False),
    "deepseek-reasoner": ("deepseek-flash", True),
    "deepseek-v4-flash": ("deepseek-flash", None),
    "deepseek-v4-flash-vision-exp": ("deepseek-flash", None),
}

REASONING_EFFORTS = ("low", "high", "max")


def detect_provider(base_url: str | None) -> str:
    """Infer the provider from an API base URL."""
    url = (base_url or "").lower()
    if "api.deepseek.com" in url:
        return DEEPSEEK
    if "openrouter.ai" in url:
        return OPENROUTER
    return OTHER


def api_key_env_var(base_url: str | None) -> str:
    """Environment variable holding the API key for ``base_url``.

    Non-DeepSeek endpoints keep using OPENROUTER_API_KEY, as before.
    """
    return PROVIDER_API_KEY_ENV.get(detect_provider(base_url), PROVIDER_API_KEY_ENV[OPENROUTER])


def api_key_from_env(base_url: str | None) -> str:
    return os.getenv(api_key_env_var(base_url)) or ""


def default_timeout(base_url: str | None) -> int:
    return DEFAULT_TIMEOUTS.get(detect_provider(base_url), FALLBACK_TIMEOUT)


def resolve_deepseek_model(model: str) -> tuple[str, bool | None]:
    """Map a model name to a current DeepSeek API model.

    Accepts OpenRouter-style slugs (``deepseek/deepseek-reasoner``) and the
    retired names. Returns ``(model, thinking)`` where ``thinking`` is the
    mode implied by a legacy name (True/False) or None when unspecified.
    """
    name = (model or "").strip()
    if name.lower().startswith("deepseek/"):
        name = name.split("/", 1)[1]
    if not name:
        return DEFAULT_MODELS[DEEPSEEK], None
    current, thinking = DEEPSEEK_LEGACY_MODELS.get(name.lower(), (name, None))
    return current, thinking


def normalize_model(model: str, base_url: str | None) -> str:
    """Adapt a model name to the provider's naming scheme."""
    if detect_provider(base_url) == DEEPSEEK:
        return resolve_deepseek_model(model)[0]
    return model


def deepseek_supports_vision(model: str) -> bool:
    return resolve_deepseek_model(model)[0] in DEEPSEEK_VISION_MODELS


def resolve_provider(provider: str | None, base_url: str | None) -> tuple[str, str]:
    """Resolve (provider, base_url) for the CLI.

    An explicit ``base_url`` always wins. ``provider="auto"`` (or None) picks
    OpenRouter when OPENROUTER_API_KEY is set, otherwise DeepSeek when
    DEEPSEEK_API_KEY is set, otherwise OpenRouter.
    """
    if base_url:
        return detect_provider(base_url), base_url
    if provider in (None, "", "auto"):
        if os.getenv(PROVIDER_API_KEY_ENV[OPENROUTER]):
            provider = OPENROUTER
        elif os.getenv(PROVIDER_API_KEY_ENV[DEEPSEEK]):
            provider = DEEPSEEK
        else:
            provider = OPENROUTER
    if provider not in PROVIDER_BASE_URLS:
        raise ValueError(
            f"Unknown provider '{provider}'. Choose from: auto, {', '.join(PROVIDER_BASE_URLS)}"
        )
    return provider, PROVIDER_BASE_URLS[provider]
