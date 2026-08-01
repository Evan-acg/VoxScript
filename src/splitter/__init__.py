from __future__ import annotations

import re
from functools import lru_cache

from src.splitter.base import SentenceSplitter
from src.splitter.cjk import CjkSplitter
from src.splitter.space import SpaceSeparatedSplitter

_DEFAULT_ABBREVIATIONS = frozenset(
    {
        "Mr",
        "Mrs",
        "Ms",
        "Dr",
        "St",
        "Sr",
        "Jr",
        "Prof",
        "Rev",
        "Gen",
        "Col",
        "Capt",
        "Sgt",
        "Lt",
        "No",
        "Vol",
        "Ch",
    }
)

_CONTEXT_ABBREVIATIONS = frozenset(
    {
        "vs",
        "e.g",
        "i.e",
        "etc",
        "Inc",
        "Ltd",
        "Co",
        "Corp",
        "a.m",
        "p.m",
        "approx",
        "est",
        "dept",
        "min",
        "max",
        "fig",
    }
)

_EXTRA_ABBREVIATIONS: dict[str, frozenset[str]] = {
    "es": frozenset({"Sr", "Sra", "Srta", "Dra", "Lic", "Ud", "Uds"}),
    "fr": frozenset({"M", "Mme", "Mlle", "Mgr", "Me"}),
    "it": frozenset({"Sig", "Sig.ra", "Dott", "On"}),
    "de": frozenset({"Hr", "Fr"}),
}

_EXTRA_CONTEXT_ABBREVIATIONS: dict[str, frozenset[str]] = {
    "de": frozenset({"z.B", "u.a", "d.h"}),
}

_SPACE_LANGUAGES = frozenset(
    {
        "en",
        "es",
        "fr",
        "it",
        "de",
        "pt",
        "nl",
        "sv",
        "da",
        "no",
        "fi",
        "pl",
        "ru",
        "uk",
        "cs",
        "sk",
        "hu",
        "ro",
        "el",
        "tr",
        "id",
        "vi",
        "tl",
    }
)

_CJK_LANGUAGES = frozenset({"ja", "zh", "ko"})

_LANG_ALIASES = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "italian": "it",
    "german": "de",
    "portuguese": "pt",
    "dutch": "nl",
    "japanese": "ja",
    "chinese": "zh",
    "korean": "ko",
}

_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af]")
_HAN_RE = re.compile(r"[\u4e00-\u9fff]")


@lru_cache(maxsize=None)
def _space_splitter(code: str) -> SpaceSeparatedSplitter:
    return SpaceSeparatedSplitter(
        _DEFAULT_ABBREVIATIONS | _EXTRA_ABBREVIATIONS.get(code, frozenset()),
        _CONTEXT_ABBREVIATIONS
        | _EXTRA_CONTEXT_ABBREVIATIONS.get(code, frozenset()),
    )


@lru_cache(maxsize=None)
def _cjk_splitter() -> CjkSplitter:
    return CjkSplitter()


def get_splitter(
    language: str | None = None, text: str | None = None
) -> SentenceSplitter:
    code = _normalize(language)
    if code in _CJK_LANGUAGES:
        return _cjk_splitter()
    if code in _SPACE_LANGUAGES:
        return _space_splitter(code)
    if text and detect_language(text) is not None:
        return _cjk_splitter()
    return _space_splitter("en")


def detect_language(text: str) -> str | None:
    """Detect script family from content: ja/ko/zh, or None for Latin scripts."""
    sample = text[:2000]
    if _KANA_RE.search(sample):
        return "ja"
    if _HANGUL_RE.search(sample):
        return "ko"
    if _HAN_RE.search(sample):
        return "zh"
    return None


def _normalize(language: str | None) -> str | None:
    if not language:
        return None
    stripped = language.strip().lower()
    code = _LANG_ALIASES.get(stripped)
    if code is not None:
        return code
    if not stripped or not stripped[0].isalpha():
        return None
    return stripped.split("-")[0].split("_")[0]
