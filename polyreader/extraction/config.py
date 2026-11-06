"""Configuration for the extraction pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from polyreader.llm.providers import api_key_from_env


@dataclass
class ExtractionConfig:
    """Configuration for the extraction pipeline."""

    # LLM settings — OpenRouter by default. For the DeepSeek API use
    # base_url="https://api.deepseek.com", model="deepseek-flash" (key: DEEPSEEK_API_KEY).
    model: str = "deepseek/deepseek-reasoner"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    temperature: float = 1.0
    thinking_enabled: bool = True  # Must be explicitly enabled
    include_extra_body: bool = True
    # DeepSeek API: thinking depth (low | high | max; None = API default) and
    # an explicit switch to run without thinking.
    reasoning_effort: str | None = None
    thinking_disabled: bool = False
    # Per-request timeout in seconds (None = provider default: 600 s for the
    # DeepSeek API, 120 s otherwise).
    request_timeout: float | None = None

    # Vision (PDF page image inspection via VLM)
    vision_enabled: bool = False

    # Agent iteration limits
    entity_max_iterations: int = 12
    guided_max_iterations: int = 12
    max_concurrent_entities: int = 3
    max_entity_retries: int = 3
    max_samples: int = 0  # 0 = unlimited; >0 caps extraction tasks

    # Extraction mode: "agentic" (per-sample tool navigation, default),
    # "full_context" (whole paper in the prompt, batched extraction) or
    # "auto" (full_context when the packed paper fits full_context_budget).
    extraction_mode: str = "agentic"
    full_context_budget: int = 80_000  # tokens of packed page text
    batch_size: int = 4  # samples per full-context extraction call

    # Output
    output_dir: str = ""

    def __post_init__(self):
        if not self.api_key:
            # DEEPSEEK_API_KEY for api.deepseek.com, OPENROUTER_API_KEY otherwise.
            self.api_key = api_key_from_env(self.base_url)


# Name used before paperindex and polyextract were merged into polyreader.
PolyextractConfig = ExtractionConfig
