from __future__ import annotations

import re

from src.entity.subtitle import DialogueLine, SubtitleSegment
from src.parser.base import SubtitleParser

_TIME_RE = re.compile(
    r"^\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)
_HTML_TAG_RE = re.compile(r"<[^>]+>", re.IGNORECASE)


def _parse_timestamp(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def _split_blocks(content: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in content.splitlines():
        if not line.strip():
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(line)
    if current:
        blocks.append(current)
    return blocks


class SrtParser(SubtitleParser):
    def parse(
        self, content: str
    ) -> tuple[list[SubtitleSegment], dict[str, dict[str, str]]]:
        segments: list[SubtitleSegment] = []
        for block in _split_blocks(content):
            lines = [line.strip() for line in block]
            time_index = next(
                (i for i, line in enumerate(lines) if _TIME_RE.match(line)), -1
            )
            if time_index < 0:
                continue
            match = _TIME_RE.match(lines[time_index])
            if match is None:
                continue
            start = _parse_timestamp(*match.group(1, 2, 3, 4))
            end = _parse_timestamp(*match.group(5, 6, 7, 8))

            index: int | None = None
            if time_index > 0 and lines[time_index - 1].isdigit():
                index = int(lines[time_index - 1])

            text_lines = [
                re.sub(r"\s+", " ", _HTML_TAG_RE.sub("", line).strip())
                for line in lines[time_index + 1 :]
            ]
            text_lines = [line for line in text_lines if line]
            if text_lines:
                segments.append(
                    SubtitleSegment(
                        index=index,
                        start=start,
                        end=end,
                        text="\n".join(text_lines),
                        lines=[
                            DialogueLine(style="Default", content=line)
                            for line in text_lines
                        ],
                    )
                )
        return segments, {}
