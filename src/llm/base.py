from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class ChatResult:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMProvider(ABC):
    name: ClassVar[str]

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        timeout: float,
    ) -> ChatResult:
        """Send a chat completion request and return the assistant text plus token usage."""
