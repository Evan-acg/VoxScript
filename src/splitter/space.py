from __future__ import annotations

import re

from src.splitter.base import SentenceSplitter

_WEAK_RE = re.compile(r"(?<=[,;—])\s*")
_STRONG_RE = re.compile(r"[.!?]+(?:\s+|$)")
_NUMBER_DOT_RE = re.compile(r"(?<=\d)\.(?=\d)")
_CLAUSE_WORDS = (
    "and",
    "but",
    "so",
    "because",
    "or",
    "if",
    "when",
    "which",
    "that",
    "then",
    "while",
    "although",
    "though",
    "however",
)
_CLAUSE_PATTERNS = tuple(
    re.compile(rf"\b{re.escape(word)}\b") for word in _CLAUSE_WORDS
)
_PLACEHOLDER = "\x00"


def _compile_pattern(
    abbreviations: set[str], lookahead: str | None = None
) -> re.Pattern[str] | None:
    if not abbreviations:
        return None
    pattern = r"\b(?:" + "|".join(
        re.escape(word)
        for word in sorted(abbreviations, key=len, reverse=True)
    ) + r")\."
    if lookahead:
        pattern += lookahead
    return re.compile(pattern)


class SpaceSeparatedSplitter(SentenceSplitter):
    family = "space"
    max_chars = 80

    def __init__(
        self,
        abbreviations: set[str],
        context_abbreviations: set[str] = frozenset(),
    ) -> None:
        self._abbrev_re = _compile_pattern(abbreviations)
        self._context_abbrev_re = _compile_pattern(
            context_abbreviations, r"(?=\s*[a-z])"
        )

    def split(self, text: str) -> list[str]:
        protected = self._protect(text)
        sentences: list[str] = []
        for part in _STRONG_RE.split(protected):
            part = part.replace(_PLACEHOLDER, ".").strip()
            if not part:
                continue
            sentences.extend(self.split_long(part))
        return sentences

    def join_text(self, parts: list[str]) -> str:
        return " ".join(parts)

    def _protect(self, text: str) -> str:
        protected = _NUMBER_DOT_RE.sub(
            lambda match: match.group(0).replace(".", _PLACEHOLDER), text
        )
        for pattern in (self._abbrev_re, self._context_abbrev_re):
            if pattern is not None:
                protected = pattern.sub(
                    lambda match: match.group(0).replace(".", _PLACEHOLDER), protected
                )
        return protected

    def _weak_split(self, text: str) -> list[str]:
        return _WEAK_RE.split(text)

    def _clause_end_before(self, text: str) -> int | None:
        scope = text[: self.max_chars]
        best: int | None = None
        for pattern in _CLAUSE_PATTERNS:
            for match in pattern.finditer(scope):
                best = match.end()
        return best

    def _hard_cut_before(self, text: str) -> int | None:
        index = text[: self.max_chars].rfind(" ")
        if index < 0:
            return None
        return index + 1
