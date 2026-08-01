from __future__ import annotations

import tempfile
from datetime import datetime
from functools import cached_property
from pathlib import Path

from pydantic import BaseModel

from src.entity.subtitle import ParsedSubtitle, SubtitleSegment


class PipelineContext(BaseModel):
    work_dir: Path | None = None
    audio_path: Path | None = None
    audio_track: int | None = None
    transcript_path: Path | None = None
    transcript_normalized_path: Path | None = None
    split_json_path: Path | None = None
    normalized_paths: list[Path] = []
    user_subtitles: list[ParsedSubtitle] = []
    transcript_segments: list[SubtitleSegment] = []

    @cached_property
    def run_dir(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = self.work_dir or Path(tempfile.gettempdir())
        run_dir = base / f"Vox_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
