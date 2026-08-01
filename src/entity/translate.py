from __future__ import annotations

import re

from pydantic import BaseModel, field_validator

_TPL_RE = re.compile(r"^\{\{\s*(.+?)\s*\}\}$")


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = ""
    api_endpoint: str = ""
    api_key: str = ""
    api_key_env: str = "LLM_API_KEY"
    source_lang: str = "en"
    target_lang: str = "zh-CN"
    term_base: dict[str, str] = {}
    max_lines_per_request: int = 30
    temperature: float = 0.3
    summary: str = ""
    characters: str = ""
    relationships: str = ""
    target_style: str = "Default"
    alias_groups: list[list[str]] = []

    @field_validator("api_endpoint")
    @classmethod
    def _validate_endpoint(cls, value: str) -> str:
        if value and not value.startswith(("http://", "https://")):
            raise ValueError(f"api_endpoint must start with http:// or https://: {value}")
        return value

    @field_validator("api_key_env")
    @classmethod
    def _normalize_api_key_env(cls, value: str) -> str:
        match = _TPL_RE.match(value.strip())
        return match.group(1) if match else value

    @field_validator("max_lines_per_request")
    @classmethod
    def _validate_batch_size(cls, value: int) -> int:
        if not 1 <= value <= 100:
            raise ValueError("max_lines_per_request must be between 1 and 100")
        return value

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")
        return value
