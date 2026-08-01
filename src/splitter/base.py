from __future__ import annotations

from abc import ABC, abstractmethod


class SentenceSplitter(ABC):
    family: str
    max_chars: int

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """Split text into sentences using language-specific rules."""

    @abstractmethod
    def join_text(self, parts: list[str]) -> str:
        """Join sentence parts with the language-appropriate separator."""

    def split_long(self, text: str) -> list[str]:
        """Fallback splitting: weak separators, clause boundaries, hard cut."""
        if len(text) <= self.max_chars:
            return [text]
        tokens = [token for token in self._weak_split(text) if token]
        if len(tokens) > 1:
            return self._greedy_group(tokens)
        cut = self._clause_end_before(text) or self._hard_cut_before(text) or self.max_chars
        return self.split_long(text[:cut]) + self.split_long(text[cut:])

    def _greedy_group(self, tokens: list[str]) -> list[str]:
        parts: list[str] = []
        current = ""
        for token in tokens:
            if len(token) > self.max_chars:
                if current:
                    parts.append(current)
                    current = ""
                parts.extend(self.split_long(token))
            elif not current or len(current) + len(token) <= self.max_chars:
                current += token
            else:
                parts.append(current)
                current = token
        if current:
            parts.append(current)
        return parts

    @abstractmethod
    def _weak_split(self, text: str) -> list[str]:
        """Split text on weak separators, keeping them attached to the part."""

    @abstractmethod
    def _clause_end_before(self, text: str) -> int | None:
        """Return the position right after the last clause word in text[:max_chars]."""

    @abstractmethod
    def _hard_cut_before(self, text: str) -> int | None:
        """Return the position of the last word boundary in text[:max_chars]."""
