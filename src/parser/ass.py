from __future__ import annotations

import re

from src.entity.subtitle import DialogueLine, SubtitleSegment
from src.parser.base import SubtitleParser

_TIME_RE = re.compile(r"^(\d+):(\d{2}):(\d{2})\.(\d{1,2})")
_EVENT_RE = re.compile(r"^(Dialogue|Comment):\s*(.*)$", re.IGNORECASE)
_TAG_RE = re.compile(r"\{[^}]*\}")
_RSTYLE_RE = re.compile(r"\{\\r([^}]*)\}", re.IGNORECASE)

_REQUIRED_FIELDS = ("start", "end", "text")


def _parse_ass_timestamp(value: str) -> float | None:
    match = _TIME_RE.match(value.strip())
    if match is None:
        return None
    hours, minutes, seconds, centis = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(centis) / 100


def _clean_text(text: str) -> str:
    text = _TAG_RE.sub("", text)
    text = text.replace("\\h", " ")
    return re.sub(r"\s+", " ", text).strip()


def _parse_lines(text: str, base_style: str) -> list[DialogueLine]:
    lines: list[DialogueLine] = []
    cur_style = base_style
    for chunk in re.split(r"\\[Nn]", text):
        pos = 0
        for match in _RSTYLE_RE.finditer(chunk):
            if match.start() > pos:
                content = _clean_text(chunk[pos : match.start()])
                if content:
                    lines.append(DialogueLine(style=cur_style, content=content))
            cur_style = match.group(1) or base_style
            pos = match.end()
        if pos < len(chunk):
            content = _clean_text(chunk[pos:])
            if content:
                lines.append(DialogueLine(style=cur_style, content=content))
    return lines


class AssParser(SubtitleParser):
    def parse(
        self, content: str
    ) -> tuple[list[SubtitleSegment], dict[str, dict[str, str]]]:
        segments: list[SubtitleSegment] = []
        styles: dict[str, dict[str, str]] = {}
        section: str | None = None
        field_names: list[str] | None = None
        index = 0

        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].strip().lower()
                field_names = None
                continue
            if line.lower().startswith("format:"):
                field_names = [
                    field.strip() for field in line.split(":", 1)[1].split(",")
                ]
                continue
            if section == "v4+ styles":
                if line.lower().startswith("style:") and field_names:
                    values = [
                        value.strip()
                        for value in line.split(":", 1)[1].split(",")
                    ]
                    styles[values[0]] = dict(
                        zip(field_names, values, strict=True)
                    )
                continue
            if section != "events" or field_names is None:
                continue

            match = _EVENT_RE.match(line)
            if match is None:
                continue
            event_type, payload = match.group(1).lower(), match.group(2)
            if event_type != "dialogue":
                continue

            fields = payload.split(",", len(field_names) - 1)
            if len(fields) != len(field_names):
                continue
            event = {
                name.lower(): value.strip()
                for name, value in zip(field_names, fields, strict=True)
            }
            if not all(name in event for name in _REQUIRED_FIELDS):
                continue

            start = _parse_ass_timestamp(event["start"])
            end = _parse_ass_timestamp(event["end"])
            if start is None or end is None:
                continue

            event_style = event.get("style", "")
            lines = _parse_lines(event["text"], event_style)
            if not lines:
                continue

            index += 1
            segments.append(
                SubtitleSegment(
                    index=index,
                    start=start,
                    end=end,
                    style=event_style or None,
                    text="\n".join(line.content for line in lines),
                    lines=lines,
                )
            )
        return segments, styles
