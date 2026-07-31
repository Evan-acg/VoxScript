from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from src.entity.context import PipelineContext
from src.entity.proof import ProofArgs


class AudioHandler:
    def __init__(self, args: ProofArgs, context: PipelineContext) -> None:
        self.args = args
        self.context = context

    def extract_audio(self) -> PipelineContext:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found on PATH")

        work_dir = Path(tempfile.mkdtemp(prefix="voxscript_"))
        output_path = work_dir / f"{self.args.input_path.stem}.wav"

        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(self.args.input_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip()
            raise RuntimeError(f"ffmpeg failed: {detail or 'unknown error'}")
        self.context.audio_path = output_path
        return self.context
