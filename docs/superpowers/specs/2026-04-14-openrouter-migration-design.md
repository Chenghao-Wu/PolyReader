# OpenRouter Migration Design Spec

**Date:** 2026-04-14
**Goal:** Replace all direct LLM provider integrations with OpenRouter as a single gateway, enabling model switching via the `provider/model-name` format.

---

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Provider strategy | Replace all providers with OpenRouter | Single gateway simplifies key management and enables model switching |
| Gemini native SDK | Drop entirely | Route Gemini models through OpenRouter's OpenAI-compatible API |
| Config pattern | Config.yaml default + CLI override per run | Matches existing pattern, just pointed at OpenRouter |
| API key | Single `OPENROUTER_API_KEY` env var | Replaces 4-var chain (`ZHIPU_`, `GEMINI_`, `OPENAI_`, `PAPERINDEX_API_KEY`) |
| Thinking/extra_body | Keep toggle, keep `--no-extra-body` escape hatch | Some models support it, some don't; user controls per-run |
| Provider heuristics | Keep `_is_reasoner_model()` as-is | Detects intrinsic model property, distinct from `thinking_enabled` user flag |
| Approach | Minimal swap (Approach 1) | Smallest diff, preserves battle-tested `LLMClient`, lowest risk |

---

## Section 1: Configuration Layer

### `paperindex/config.yaml`

Change LLM defaults to OpenRouter:

```yaml
model: "deepseek/deepseek-chat"
base_url: "https://openrouter.ai/api/v1"
temperature: 1
max_concurrent_api_calls: 5
include_extra_body: true
```

### `paperindex/config.py`

Replace the 4-env-var chain (lines 44-51) with a single lookup:

```python
if os.getenv("OPENROUTER_API_KEY"):
    merged.setdefault("api_key", os.getenv("OPENROUTER_API_KEY"))
```

Update error message to reference `OPENROUTER_API_KEY`.

### `polyextract/config.py`

- Default `model` → `"deepseek/deepseek-reasoner"`
- Default `base_url` → `"https://openrouter.ai/api/v1"`
- `__post_init__` env var chain → `os.getenv("OPENROUTER_API_KEY") or ""`

### `polyextract/cli.py`

- `--model` default → `"deepseek/deepseek-reasoner"`
- `--base-url` default → `"https://openrouter.ai/api/v1"`
- Keep `--no-extra-body` flag (escape hatch for models that reject it)
- Remove Gemini auto-detection block (lines 388-390)

**Note on default models:** `paperindex/config.yaml` defaults to `deepseek/deepseek-chat` (cheaper, for indexing/retrieval). `polyextract/cli.py` defaults to `deepseek/deepseek-reasoner` (reasoning, for extraction). This is intentional — each pipeline uses the model tier appropriate for its task.

---

## Section 2: LLM Client Changes

### `llm/__init__.py` — Factory simplification

Remove the Gemini branch. Factory always returns `LLMClient`:

```python
def create_llm_client(
    api_key: str,
    model: str,
    base_url: str = "https://openrouter.ai/api/v1",
    temperature: float = 1.0,
    timeout: int = 120,
    max_concurrent: int = 1,
    thinking_enabled: bool = False,
    include_extra_body: bool = True,
):
    """Create an OpenRouter-backed LLM client."""
    return LLMClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        timeout=timeout,
        max_concurrent=max_concurrent,
        thinking_enabled=thinking_enabled,
        include_extra_body=include_extra_body,
    )
```

### `llm/client.py` — OpenRouter headers + model slug warning

Add OpenRouter identification headers in `_get_or_create_client()`:

```python
self._client = openai.AsyncOpenAI(
    api_key=self.api_key,
    base_url=self.base_url,
    timeout=self.timeout,
    max_retries=0,
    default_headers={
        "HTTP-Referer": "https://github.com/paperindex",
        "X-Title": "paperindex",
    },
)
```

Add model slug format warning in `__init__`:

```python
if "/" not in self.model:
    logger.warning(
        f"Model '{self.model}' has no provider prefix. "
        f"OpenRouter expects 'provider/model-name' format "
        f"(e.g. 'deepseek/deepseek-reasoner')."
    )
```

Soft warning, not an error — preserves `--base-url` override with non-OpenRouter endpoints.

### `llm/gemini_client.py` — Delete

Remove the entire file. All Gemini models route through OpenRouter.

---

## Section 3: Provider Heuristics — No Change

The existing heuristics are correct and survive OpenRouter model slugs:

| Heuristic | What it does | Why it's safe |
|-----------|-------------|---------------|
| `_is_reasoner_model()` | `"reasoner" in self.model.lower()` | Detects intrinsic model property (always-on reasoning, rejects tool messages). Works with OpenRouter slugs: `"reasoner" in "deepseek/deepseek-reasoner"` → True |
| Instructor mode | MD_JSON for reasoner, TOOLS otherwise | Driven by model property, not user flag. Correct separation. |
| `strip_tool_messages()` | Strips tool messages when reasoner + tool_choice=="none" | Harmless workaround, still needed for DeepSeek reasoner through OpenRouter |
| `_build_extra_body()` | Sends thinking params when `thinking_enabled=True` | Already uses user flag correctly. Orthogonal to `_is_reasoner_model()`. |

**Key distinction (add as docstring):** `_is_reasoner_model()` detects an intrinsic model property. `thinking_enabled` is a user opt-in for the `extra_body` API feature. They are orthogonal and must not be conflated.

---

## Section 4: Caller-Side Cleanup

### `paperindex/__init__.py`

- Delete Gemini auto-detection block (lines 254-257): `if "gemini" in opt.model.lower() and not base_url: opt.base_url = ""`
- Update error message (line 261-263) to reference `OPENROUTER_API_KEY`

### `polyextract/cli.py`

- Delete Gemini auto-detection block (lines 388-390)

### `pyproject.toml`

- Remove `google-genai>=1.0` dependency (no longer needed without `gemini_client.py`)

---

## Section 5: Explicit Scope Boundaries

**NOT changing:**

- `strip_tool_messages()` — still needed for DeepSeek reasoner
- `clear_reasoning_content()` — DeepSeek reasoner still produces reasoning_content
- `chat_vision()` — OpenRouter supports vision via OpenAI image_url format
- `chat_with_tools()` — OpenRouter supports function calling via OpenAI tool format
- `chat_structured()` / Instructor — works against any OpenAI-compatible endpoint
- `--base-url` CLI flag — escape valve for direct provider access
- `--no-extra-body` CLI flag — escape valve for models that reject extra_body
- `--index-model` CLI flag — still useful for mixed model strategies
- `llm/token_counter.py` — no provider dependency
- `polyextract/orchestrator.py` — calls `create_llm_client()`, interface unchanged

---

## Section 6: Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| OpenRouter drops/mangles `extra_body` for thinking | Medium | `--no-extra-body` flag; test with deepseek-reasoner on first run |
| User passes bare model name (old habit) | High initially | Soft warning in logs; not a hard error |
| Latency increase from proxy hop | Certain but small | `--base-url` override for direct access when needed |
| OpenRouter pricing markup | Certain | Accepted trade-off; `--base-url` as escape valve |
| Model-specific quirks undiscovered | Low | Retry logic + `--no-extra-body` + `--base-url` overrides |

---

## Files Changed Summary

| File | Action |
|------|--------|
| `paperindex/config.yaml` | Edit: model, base_url defaults |
| `paperindex/config.py` | Edit: env var chain → single OPENROUTER_API_KEY |
| `paperindex/__init__.py` | Edit: remove Gemini detection, update error msg |
| `paperindex/llm/__init__.py` | Edit: remove Gemini branch, update default base_url |
| `paperindex/llm/client.py` | Edit: add OpenRouter headers, slug warning, docstring |
| `paperindex/llm/gemini_client.py` | Delete |
| `polyextract/config.py` | Edit: model, base_url, env var defaults |
| `polyextract/cli.py` | Edit: model/base-url defaults, remove Gemini detection |
| `pyproject.toml` | Edit: remove google-genai dependency |
