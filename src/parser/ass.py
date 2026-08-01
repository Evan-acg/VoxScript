from __future__ import annotations

import re

from src.entity.subtitle import SubtitleSegment
from src.parser.base import SubtitleParser

_TIME_RE = re.compile(r"^(\d+):(\d{2}):(\d{2})\.(\d{1,2})")
_EVENT_RE = re.compile(r"^(Dialogue|Comment):\s*(.*)$", re.IGNORECASE)
_OVERRIDE_TAG_RE = re.compile(r"\{[^}]*\}")

_REQUIRED_FIELDS = ("start", "end", "text")


def _parse_ass_timestamp(value: str) -> float | None:
    match = _TIME_RE.match(value.strip())
    if match is None:
        return None
    hours, minutes, seconds, centis = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(centis) / 100


def _strip_override_tags(text: str) -> str:
    text = _OVERRIDE_TAG_RE.sub("", text)
    text = (
        text.replace("\\N", " ")
        .replace("\\n", " ")
        .replace("\\h", " ")
        .replace("\\t", " ")
    )
    return re.sub(r"\s+", " ", text).strip()


class AssParser(SubtitleParser):
    def parse(self, content: str) -> list[SubtitleSegment]:
        segments: list[SubtitleSegment] = []
        in_events = False
        field_names: list[str] | None = None
        index = 0

        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                in_events = line[1:-1].strip().lower() == "events"
                field_names = None
                continue
            if not in_events:
                continue
            if line.lower().startswith("format:"):
                field_names = [field.strip() for field in line.split(":", 1)[1].split(",")]
                continue

            match = _EVENT_RE.match(line)
            if match is None or field_names is None:
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
            text = _strip_override_tags(event["text"])
            if start is None or end is None or not text:
                continue

            index += 1
            segments.append(
                SubtitleSegment(index=index, start=start, end=end, text=text)
            )
        return segments
