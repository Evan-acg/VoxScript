from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..config import get, get_int
from ..progress import ProgressCallback, ProgressEvent, null_callback


class MediaError(Exception):
    pass


def build_extract_command(
    ffmpeg_path: str,
    input_path: str,
    output_path: str,
    *,
    stream_index: int,
    start: float,
    end: float,
    sample_rate: int,
    channels: int,
) -> list[str]:
    if end <= start:
        raise ValueError("end must be greater than start")
    return [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(start),
        "-i",
        input_path,
        "-map",
        f"0:{stream_index}",
        "-vn",
        "-acodec",
        get("audio", "codec", fallback="pcm_s16le"),
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-t",
        str(end - start),
        "-y",
        "-progress",
        "pipe:1",
        "-nostats",
        output_path,
    ]


class FFmpegAudioExtractor:
    def __init__(self, ffmpeg_path: str = "", ffprobe_path: str = "") -> None:
        self._ffmpeg_path = ffmpeg_path or get("ffmpeg", "path", fallback="ffmpeg")
        self._ffprobe_path = ffprobe_path or get("ffmpeg", "probe_path", fallback="ffprobe")

    def get_duration(self, input_path: str) -> float:
        command = [
            self._ffprobe_path,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            input_path,
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            return float(json.loads(result.stdout)["format"]["duration"])
        except (FileNotFoundError, subprocess.CalledProcessError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise MediaError(f"failed to read media duration: {error}") from error

    def list_audio_streams(self, input_path: str) -> list[int]:
        command = [
            self._ffprobe_path,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-select_streams",
            "a",
            input_path,
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            streams = json.loads(result.stdout).get("streams", [])
            return [int(stream["index"]) for stream in streams]
        except (FileNotFoundError, subprocess.CalledProcessError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise MediaError(f"failed to list audio streams: {error}") from error

    def extract(
        self,
        input_path: str,
        output_path: str,
        *,
        stream_index: int | None,
        start: float,
        end: float,
        force: bool,
        on_progress: ProgressCallback = null_callback,
    ) -> None:
        output = Path(output_path)
        if output.exists() and not force:
            on_progress(ProgressEvent("ffmpeg", 1, 1, "Using cached chunk audio"))
            return

        streams = self.list_audio_streams(input_path)
        if not streams:
            raise MediaError("no audio streams found")
        selected_stream = stream_index if stream_index is not None else streams[0]
        if selected_stream not in streams:
            raise MediaError(
                f"audio stream #{selected_stream} not found; available: {', '.join(map(str, streams))}"
            )

        sample_rate = get_int("audio", "sample_rate", fallback=16000)
        channels = get_int("audio", "channels", fallback=1)
        command = build_extract_command(
            self._ffmpeg_path,
            input_path,
            str(output),
            stream_index=selected_stream,
            start=start,
            end=end,
            sample_rate=sample_rate,
            channels=channels,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        duration = end - start
        on_progress(ProgressEvent("ffmpeg", 0, duration, "Extracting chunk audio"))
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
            for line in process.stdout or ():
                if line.startswith("out_time="):
                    current = _parse_ffmpeg_time(line.removeprefix("out_time="))
                    on_progress(ProgressEvent("ffmpeg", current, duration, "Extracting chunk audio"))
            stderr = process.stderr.read() if process.stderr else ""
            return_code = process.wait()
        except FileNotFoundError as error:
            raise MediaError(f"ffmpeg not found: {error}") from error
        if return_code != 0:
            raise MediaError(f"ffmpeg failed with code {return_code}: {stderr.strip()}")
        on_progress(ProgressEvent("ffmpeg", duration, duration, "Chunk audio extracted"))


def _parse_ffmpeg_time(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) != 3:
        return 0.0
    try:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except ValueError:
        return 0.0
