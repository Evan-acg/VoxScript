from __future__ import annotations

import re
from pathlib import Path

from src.entity.subtitle import (
    DialogueLine,
    NormalizedDialogue,
    NormalizedSubtitle,
    ParsedSubtitle,
    SubtitleSegment,
)
from src.parser import parse_subtitle


def parse_subtitle_file(path: Path, label: str = "subtitle") -> ParsedSubtitle:
    try:
        return parse_subtitle(path)
    except (ValueError, RuntimeError) as exc:
        raise RuntimeError(f"failed to parse {label}: {path} - {exc}") from exc


def write_normalized_json(parsed: ParsedSubtitle, output_path: Path) -> None:
    dialogue = [
        NormalizedDialogue(
            index=index,
            start=segment.start,
            end=segment.end,
            lines=_segment_lines(segment),
        )
        for index, segment in enumerate(parsed.segments, start=1)
    ]
    normalized = NormalizedSubtitle(
        path=parsed.path,
        format=parsed.format,
        styles=parsed.styles,
        dialogue=dialogue,
    )
    text = normalized.model_dump_json(indent=2, ensure_ascii=False) + "\n"
    output_path.write_text(text, encoding="utf-8")


def _segment_lines(segment: SubtitleSegment) -> list[DialogueLine]:
    if segment.lines is not None:
        return [line for line in segment.lines if line.content]
    text_lines = [line.strip() for line in segment.text.split("\n") if line.strip()]
    return [
        DialogueLine(style="Default", content=re.sub(r"\s+", " ", line))
        for line in text_lines
    ]
