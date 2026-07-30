from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SubtitleEvent:
    index: int
    start: float
    end: float
    text: str


@dataclass
class SubtitleDocument:
    format: str
    header: str = ""
    events: list[SubtitleEvent] = field(default_factory=list)


def _parse_srt_time(t: str) -> float:
    h, m, rest = t.split(":")
    s_str = rest.replace(",", ".")
    return int(h) * 3600 + int(m) * 60 + float(s_str)


def _format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def _is_ass_format(content: str) -> bool:
    return bool(re.search(r"^\[V4\+? Styles\]", content, re.MULTILINE))


def _is_ssa_format(content: str) -> bool:
    return bool(re.search(r"^\[V4 Styles\]", content, re.MULTILINE))


def parse_srt(content: str) -> SubtitleDocument:
    blocks = re.split(r"\n\n+", content.strip())
    events: list[SubtitleEvent] = []

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue
        time_range = lines[1].strip()
        text = "\n".join(lines[2:])
        start_str, end_str = time_range.split(" --> ")
        events.append(
            SubtitleEvent(
                index=idx,
                start=_parse_srt_time(start_str),
                end=_parse_srt_time(end_str),
                text=text,
            )
        )

    return SubtitleDocument(format="srt", events=events)


def parse_ass(content: str) -> SubtitleDocument:
    events: list[SubtitleEvent] = []

    header_lines: list[str] = []
    in_events = False
    in_header = True
    event_format: list[str] = []

    for line in content.splitlines():
        stripped = line.strip()

        if stripped == "[Events]":
            in_events = True
            in_header = False

        if in_header:
            header_lines.append(line)
            continue

        if in_events:
            if stripped.startswith("Format:"):
                event_format = [f.strip() for f in stripped[len("Format:"):].split(",")]
                header_lines.append(line)
                continue
            if stripped.startswith("Dialogue:"):
                parts = [p.strip() for p in stripped[len("Dialogue:"):].split(",", len(event_format) - 1)]
                if len(parts) >= len(event_format):
                    field_map = dict(zip(event_format, parts))
                    start_str = field_map.get("Start", "")
                    end_str = field_map.get("End", "")
                    text = field_map.get("Text", "")
                    l1 = field_map.get("Layer", "0")
                    try:
                        idx = int(l1) if l1 else 0
                    except ValueError:
                        idx = 0
                    if start_str and end_str:
                        events.append(
                            SubtitleEvent(
                                index=idx,
                                start=_parse_srt_time(start_str.replace(".", ",")),
                                end=_parse_srt_time(end_str.replace(".", ",")),
                                text=text,
                            )
                        )
                    continue

        header_lines.append(line)

    header = "\n".join(header_lines)
    return SubtitleDocument(format="ass" if _is_ass_format(content) else "ssa", header=header, events=events)


def parse_subtitle(path: str | Path) -> SubtitleDocument:
    path = Path(path)
    content = path.read_text(encoding="utf-8")

    if _is_ass_format(content) or _is_ssa_format(content):
        return parse_ass(content)

    return parse_srt(content)
