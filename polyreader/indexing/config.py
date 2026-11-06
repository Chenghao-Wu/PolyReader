"""Configuration loader: YAML defaults + env vars + runtime overrides."""

import os
import yaml
from pathlib import Path
from types import SimpleNamespace as config
from dotenv import load_dotenv

from polyreader.llm.providers import api_key_env_var

load_dotenv()


class ConfigLoader:
    """Load config from YAML, override with env vars and runtime args."""

    def __init__(self, default_path: str = None):
        if default_path is None:
            default_path = Path(__file__).parent / "config.yaml"
        self._default_dict = self._load_yaml(default_path)

    @staticmethod
    def _load_yaml(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _validate_keys(self, user_dict):
        unknown_keys = set(user_dict) - set(self._default_dict)
        if unknown_keys:
            raise ValueError(f"Unknown config keys: {unknown_keys}")

    def load(self, user_opt=None) -> config:
        if user_opt is None:
            user_dict = {}
        elif isinstance(user_opt, config):
            user_dict = vars(user_opt)
        elif isinstance(user_opt, dict):
            user_dict = user_opt
        else:
            raise TypeError("user_opt must be dict, config(SimpleNamespace) or None")

        self._validate_keys(user_dict)
        merged = {**self._default_dict, **user_dict}

        # Nested sections (e.g. mineru) merge key-by-key over the defaults.
        for key, default in self._default_dict.items():
            if isinstance(default, dict) and isinstance(user_dict.get(key), dict):
                merged[key] = {**default, **user_dict[key]}

        # Env var override (caller-supplied api_key wins). The variable
        # depends on the endpoint: DEEPSEEK_API_KEY for api.deepseek.com,
        # OPENROUTER_API_KEY otherwise.
        if key := os.getenv(api_key_env_var(merged.get("base_url"))):
            merged.setdefault("api_key", key)

        return config(**merged)
