from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from src.core.events import EventBus
from src.entity.context import PipelineContext
from src.entity.proof import ProofArgs

TrackSelector = Callable[[list[dict]], int]


class AudioHandler:
    def __init__(
        self,
        args: ProofArgs,
        context: PipelineContext,
        bus: EventBus,
        track_selector: TrackSelector | None = None,
    ) -> None:
        self.args = args
        self.context = context
        self.bus = bus
        self.track_selector = track_selector

    def extract_audio(self) -> PipelineContext:
        self.bus.log("probing audio streams...")
        streams = self._probe_audio_streams()
        stream_index = self._select_audio_stream(streams)

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found on PATH")

        work_dir = Path(tempfile.mkdtemp(prefix="voxscript_"))
        output_path = work_dir / f"{self.args.input_path.stem}.wav"
        duration = self._probe_duration()

        self.bus.log(f"extracting audio track {stream_index} ...")
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(self.args.input_path),
            "-map",
            f"0:{stream_index}",
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_path),
            "-progress",
            "pipe:1",
            "-nostats",
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.stdout is not None:
            for line in proc.stdout:
                self._parse_progress(line, duration)
        stderr = proc.stderr.read() if proc.stderr is not None else ""
        returncode = proc.wait()

        if returncode != 0:
            detail = stderr.strip()
            message = f"ffmpeg failed: {detail or 'unknown error'}"
            self.bus.log(message, level="ERROR")
            raise RuntimeError(message)

        self.bus.set_progress("extract_audio", 100.0)
        self.bus.log("audio extraction complete")

        self.context.audio_path = output_path
        self.context.audio_track = stream_index
        return self.context

    def _probe_audio_streams(self) -> list[dict]:
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            raise RuntimeError("ffprobe not found on PATH")

        cmd = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index,codec_name,channels:stream_tags=language",
            "-of",
            "json",
            str(self.args.input_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip()
            raise RuntimeError(f"ffprobe failed: {detail or 'unknown error'}")
        data = json.loads(proc.stdout or "{}")
        return data.get("streams", [])

    def _probe_duration(self) -> float | None:
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            return None
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(self.args.input_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return None
        try:
            return float(proc.stdout.strip())
        except ValueError:
            return None

    def _parse_progress(self, line: str, duration: float | None) -> None:
        if "=" not in line or not duration:
            return
        key, _, value = line.strip().partition("=")
        if key == "out_time_us":
            try:
                us = int(value)
            except ValueError:
                return
            pct = min(100.0, max(0.0, us / 1_000_000 / duration * 100))
            self.bus.set_progress("extract_audio", pct)

    def _select_audio_stream(self, streams: list[dict]) -> int:
        if not streams:
            raise RuntimeError("no audio track found in input file")

        if len(streams) == 1:
            index = streams[0]["index"]
            self.bus.log(f"1 audio track found, auto-selected index {index}")
            return index

        self.bus.log(f"{len(streams)} audio tracks found, prompting for selection")
        if self.track_selector is None:
            raise RuntimeError(
                "multiple audio tracks found but no track selector available"
            )
        return self.track_selector(streams)
