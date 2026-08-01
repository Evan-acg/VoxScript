from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class SubtitleSegment(BaseModel):
    index: int | None = None
    start: float
    end: float
    text: str


class ParsedSubtitle(BaseModel):
    path: Path
    format: str
    segments: list[SubtitleSegment]
