from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable

StreamSelector = Callable[[list[dict]], int]


def probe_audio_streams(input_path: Path) -> list[dict]:
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
        str(input_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"ffprobe failed: {detail or 'unknown error'}")
    data = json.loads(proc.stdout or "{}")
    return data.get("streams", [])


def probe_duration(input_path: Path) -> float | None:
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
        str(input_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def select_audio_stream(
    streams: list[dict],
    prompt: StreamSelector | None = None,
) -> int:
    if not streams:
        raise RuntimeError("no audio track found in input file")

    if len(streams) == 1:
        return streams[0]["index"]

    if prompt is None:
        raise RuntimeError("multiple audio tracks found but no track selector available")
    return prompt(streams)


def extract_audio(
    input_path: Path,
    output_path: Path,
    stream_index: int,
    on_progress: Callable[[float], None] | None = None,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH")

    duration = probe_duration(input_path)
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
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
            _parse_progress(line, duration, on_progress)
    stderr = proc.stderr.read() if proc.stderr is not None else ""
    returncode = proc.wait()

    if returncode != 0:
        detail = stderr.strip()
        raise RuntimeError(f"ffmpeg failed: {detail or 'unknown error'}")


def _parse_progress(
    line: str,
    duration: float | None,
    on_progress: Callable[[float], None] | None,
) -> None:
    if "=" not in line or not duration:
        return
    key, _, value = line.strip().partition("=")
    if key == "out_time_us":
        try:
            us = int(value)
        except ValueError:
            return
        pct = min(100.0, max(0.0, us / 1_000_000 / duration * 100))
        if on_progress is not None:
            on_progress(pct)
