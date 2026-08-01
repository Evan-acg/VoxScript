from __future__ import annotations

import re

from src.splitter.base import SentenceSplitter

_STRONG_RE = re.compile(r"[。？！]+")
_WEAK_RE = re.compile(r"(?<=[、，・])")
_CLAUSE_WORDS = (
    "因为",
    "所以",
    "但是",
    "然而",
    "于是",
    "可是",
    "因此",
    "虽然",
    "如果",
    "只要",
    "しかし",
    "だから",
    "でも",
    "それで",
    "そして",
    "ところが",
    "なぜなら",
    "つまり",
)
_CLAUSE_PATTERNS = tuple(re.compile(re.escape(word)) for word in _CLAUSE_WORDS)


class CjkSplitter(SentenceSplitter):
    max_chars = 30

    def split(self, text: str) -> list[str]:
        sentences: list[str] = []
        for part in _STRONG_RE.split(text):
            part = part.strip()
            if not part:
                continue
            sentences.extend(self.split_long(part))
        return sentences

    def join_text(self, parts: list[str]) -> str:
        return "".join(parts)

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
        return None
