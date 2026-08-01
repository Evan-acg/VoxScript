from __future__ import annotations

from pathlib import Path

from src.entity.subtitle import ParsedSubtitle
from src.parser import parse_subtitle


def parse_subtitle_file(path: Path, label: str = "subtitle") -> ParsedSubtitle:
    try:
        return parse_subtitle(path)
    except (ValueError, RuntimeError) as exc:
        raise RuntimeError(f"failed to parse {label}: {path} - {exc}") from exc
