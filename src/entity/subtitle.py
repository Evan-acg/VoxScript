from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class DialogueLine(BaseModel):
    style: str
    content: str


class SubtitleSegment(BaseModel):
    index: int | None = None
    start: float
    end: float
    style: str | None = None
    text: str = ""
    lines: list[DialogueLine] | None = None


class ParsedSubtitle(BaseModel):
    path: Path
    format: str
    styles: dict[str, dict[str, str]] = {}
    segments: list[SubtitleSegment]


class NormalizedDialogue(BaseModel):
    index: int
    start: float
    end: float
    lines: list[DialogueLine]


class NormalizedSubtitle(BaseModel):
    path: Path
    format: str
    styles: dict[str, dict[str, str]]
    dialogue: list[NormalizedDialogue]
