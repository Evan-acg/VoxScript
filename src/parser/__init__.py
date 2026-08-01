from __future__ import annotations

from pathlib import Path

from src.entity.subtitle import ParsedSubtitle
from src.parser.ass import AssParser
from src.parser.base import SubtitleParser
from src.parser.srt import SrtParser

_PARSERS: dict[str, SubtitleParser] = {
    "srt": SrtParser(),
    "ass": AssParser(),
    "ssa": AssParser(),
}


def get_parser(subtitle_format: str) -> SubtitleParser:
    try:
        return _PARSERS[subtitle_format.lower()]
    except KeyError:
        raise ValueError(
            f"unsupported subtitle format: {subtitle_format}; "
            f"expected one of: {', '.join(sorted(_PARSERS))}"
        ) from None


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(
        f"cannot decode subtitle file (tried utf-8 and gb18030): {path}"
    )


def parse_subtitle(path: Path) -> ParsedSubtitle:
    subtitle_format = path.suffix.lstrip(".").lower()
    parser = get_parser(subtitle_format)
    try:
        content = _read_text(path)
    except OSError as exc:
        raise RuntimeError(f"cannot read subtitle file: {path}") from exc
    segments = parser.parse(content)
    return ParsedSubtitle(path=path, format=subtitle_format, segments=segments)
