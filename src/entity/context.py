from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from src.entity.subtitle import ParsedSubtitle, SubtitleSegment


class PipelineContext(BaseModel):
    audio_path: Path | None = None
    audio_track: int | None = None
    transcript_path: Path | None = None
    user_subtitles: list[ParsedSubtitle] = []
    transcript_segments: list[SubtitleSegment] = []
