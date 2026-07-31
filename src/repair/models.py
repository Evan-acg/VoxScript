from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AsrSegment:
    id: int
    start: float
    end: float
    text: str
    chunk_id: int | None = None


@dataclass(frozen=True)
class SubtitleCue:
    id: int
    event_index: int
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class ReviewOperation:
    action: str
    subtitle_ids: tuple[int, ...]
    asr_ids: tuple[int, ...]
    text: str
    reason: str


@dataclass(frozen=True)
class Chunk:
    id: int
    body_start: float
    body_end: float
    context_start: float
    context_end: float


@dataclass(frozen=True)
class SubtitleMatch:
    subtitle_id: int
    asr_ids: tuple[int, ...]
    score: float


@dataclass
class ApplyReport:
    applied: list[int | str]
    unresolved: list[ReviewOperation]
