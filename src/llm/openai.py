from __future__ import annotations

import requests

from src.llm.base import ChatResult, LLMProvider

_CHAT_COMPLETIONS_PATH = "/chat/completions"


def _normalize_endpoint(api_endpoint: str) -> str:
    endpoint = api_endpoint.rstrip("/")
    if endpoint.lower().endswith(_CHAT_COMPLETIONS_PATH):
        return endpoint
    return endpoint + _CHAT_COMPLETIONS_PATH


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_endpoint: str, api_key: str) -> None:
        self._api_endpoint = _normalize_endpoint(api_endpoint)
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        timeout: float,
        max_tokens: int | None = None,
    ) -> ChatResult:
        payload: dict[str, object] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        try:
            response = self._session.post(
                self._api_endpoint,
                json=payload,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        if response.status_code != 200:
            detail = response.text[:300]
            raise RuntimeError(
                f"LLM API error (HTTP {response.status_code}): {detail}"
            )
        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected LLM response format: {response.text[:300]}") from exc
        usage = data.get("usage", {})
        return ChatResult(
            content=str(content),
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        )
