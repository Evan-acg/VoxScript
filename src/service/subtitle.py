from __future__ import annotations

from pathlib import Path

from src.entity.subtitle import ParsedSubtitle, SubtitleSegment
from src.parser import parse_subtitle
from src.service.asr import format_timestamp


def parse_subtitle_file(path: Path, label: str = "subtitle") -> ParsedSubtitle:
    try:
        return parse_subtitle(path)
    except (ValueError, RuntimeError) as exc:
        raise RuntimeError(f"failed to parse {label}: {path} - {exc}") from exc


def write_srt(segments: list[SubtitleSegment], output_path: Path) -> None:
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        start = format_timestamp(segment.start)
        end = format_timestamp(segment.end)
        lines.append(str(index))
        lines.append(f"{start} --> {end}")
        lines.append(segment.text.strip())
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
