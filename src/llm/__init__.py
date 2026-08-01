from __future__ import annotations

from src.entity.translate import LLMConfig
from src.llm.base import LLMProvider
from src.llm.deepseek import DeepSeekProvider
from src.llm.openai import OpenAIProvider

_PROVIDERS: dict[str, type[LLMProvider]] = {
    OpenAIProvider.name: OpenAIProvider,
    DeepSeekProvider.name: DeepSeekProvider,
}


def create_provider(config: LLMConfig, api_key: str) -> LLMProvider:
    try:
        provider_type = _PROVIDERS[config.provider]
    except KeyError:
        raise ValueError(
            f"unsupported LLM provider: {config.provider}; "
            f"expected one of: {', '.join(sorted(_PROVIDERS))}"
        ) from None
    return provider_type(config.api_endpoint, api_key)
