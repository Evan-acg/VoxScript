from __future__ import annotations

from ..config import get
from .subtitle_parser import SubtitleDocument, SubtitleEvent


def _srt_time_to_ass(t: str) -> str:
    t = t.replace(",", ".")
    parts = t.split(":")
    hours = str(int(parts[0]))
    sec_parts = parts[2].split(".")
    seconds = sec_parts[0]
    cs = sec_parts[1][:2].ljust(2, "0") if len(sec_parts) > 1 else "00"
    return f"{hours}:{parts[1]}:{seconds}.{cs}"


def _escape_ass(text: str) -> str:
    text = text.replace("\\", "\\\\")
    text = text.replace("{", "\\{")
    text = text.replace("}", "\\}")
    return text


def _gen_dialogue(style: str, start: float, end: float, text: str) -> str:
    from .subtitle_parser import _format_srt_time

    start_str = _srt_time_to_ass(_format_srt_time(start))
    end_str = _srt_time_to_ass(_format_srt_time(end))
    escaped = _escape_ass(text)
    template = get("ass", "dialogue_template")
    if not template:
        template = "Dialogue: 2,{start},{end},{style},,0,0,0,,{text}"
    return template.format(start=start_str, end=end_str, style=style, text=escaped)


def format_ass(doc: SubtitleDocument, style: str = "Default") -> str:
    header = get("ass", "header")
    events_fmt = get("ass", "events_format")
    if not events_fmt:
        events_fmt = "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"

    lines: list[str] = [
        header,
        "",
        "[Events]",
        events_fmt,
    ]

    for event in doc.events:
        lines.append(_gen_dialogue(style, event.start, event.end, event.text))

    return "\n".join(lines) + "\n"
