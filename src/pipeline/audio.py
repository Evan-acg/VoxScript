from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from loguru import logger

from ..config import get, get_int, get_list
from ..progress import ProgressCallback, ProgressEvent, null_callback
from . import AudioStream, MediaInfo


class AudioExtractionError(Exception):
    pass


def _cache_dir() -> Path:
    d = get("audio", "cache_dir")
    if d:
        return Path(d)
    return Path(tempfile.gettempdir()) / "voxscript" / "audio"


class FFmpegAudioExtractor:
    def __init__(
        self,
        ffmpeg_path: str = "",
        ffprobe_path: str = "",
    ) -> None:
        self._ffmpeg_path = ffmpeg_path or get("ffmpeg", "path", fallback="ffmpeg")
        self._ffprobe_path = ffprobe_path or get("ffmpeg", "probe_path", fallback="ffprobe")

    def list_audio_streams(self, video_path: str) -> list[AudioStream]:
        cmd = [
            self._ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-select_streams", "a",
            video_path,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, encoding="utf-8"
            )
            data = json.loads(result.stdout)
        except (
            subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError
        ) as e:
            raise AudioExtractionError(f"Failed to probe audio streams: {e}") from e

        streams: list[AudioStream] = []
        for s in data.get("streams", []):
            tags = s.get("tags", {}) or {}
            streams.append(AudioStream(
                index=s["index"],
                codec=s.get("codec_name", "?"),
                language=tags.get("language"),
                sample_rate=int(s.get("sample_rate", 0)),
                channels=s.get("channels", 0),
                title=tags.get("title"),
            ))
        return streams

    def get_duration(self, video_path: str) -> float:
        cmd = [
            self._ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            video_path,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, encoding="utf-8"
            )
            info = json.loads(result.stdout)
            return float(info["format"]["duration"])
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, ValueError) as e:
            raise AudioExtractionError(
                f"Failed to get media duration: {e}"
            ) from e

    def extract(
        self,
        video_path: str,
        output_path: str,
        *,
        stream_index: int | None = None,
        force: bool = False,
        start: float | None = None,
        end: float | None = None,
        on_progress: ProgressCallback = null_callback,
    ) -> MediaInfo:
        resolved_idx = self._resolve_index(video_path, stream_index)

        cache_key = self._cache_key(video_path, resolved_idx, start=start, end=end)
        cache_dir = _cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cache_key}.wav"

        if not force and cache_path.exists():
            on_progress(ProgressEvent("cache", 1, 1, "Using cached audio"))
            shutil.copy2(str(cache_path), str(output_path))
            duration = self.get_duration(video_path)
            return MediaInfo(duration=duration, path=str(output_path))

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        duration = self.get_duration(video_path)

        effective = (end or duration) - (start or 0)
        on_progress(ProgressEvent("ffprobe", 1, 1, "Video info retrieved"))

        cmd = [
            self._ffmpeg_path,
        ]
        if start is not None:
            cmd.extend(["-ss", str(start)])
        cmd.extend([
            "-i", video_path,
            "-map", f"0:{resolved_idx}",
            "-vn",
            "-acodec", get("audio", "codec", fallback="pcm_s16le"),
            "-ar", str(get_int("audio", "sample_rate", fallback=16000)),
            "-ac", str(get_int("audio", "channels", fallback=1)),
        ])
        if end is not None:
            cmd.extend(["-to", str(end)])
        cmd.extend(["-y", "-progress", "pipe:", "-nostats", str(output)])

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
            )
            _track_ffmpeg_progress(proc, effective, on_progress)

            proc.wait()
            if proc.returncode != 0:
                raise AudioExtractionError(
                    f"ffmpeg exited with code {proc.returncode}"
                )
        except FileNotFoundError as e:
            raise AudioExtractionError(
                f"ffmpeg not found: {e}"
            ) from e

        shutil.copy2(str(output), str(cache_path))
        on_progress(ProgressEvent("ffmpeg", effective, effective, "Audio extracted"))
        size_mb = output.stat().st_size / (1024 * 1024)
        logger.info("Extracted audio: {:.1f}s, {:.1f}MB \u2192 {}", effective, size_mb, output.name)
        return MediaInfo(duration=duration, path=str(output))

    def _resolve_index(self, video_path: str, stream_index: int | None) -> int:
        streams = self.list_audio_streams(video_path)

        if not streams:
            raise AudioExtractionError("No audio streams found in video")

        if stream_index is not None:
            if not any(s.index == stream_index for s in streams):
                raise AudioExtractionError(
                    f"Audio stream #{stream_index} not found. "
                    f"Available: {', '.join(f'#{s.index}' for s in streams)}"
                )
            selected = next(s for s in streams if s.index == stream_index)
            self._log_stream(selected)
            return selected.index

        if len(streams) == 1:
            self._log_stream(streams[0])
            return streams[0].index

        selected = _interactive_select(streams)
        self._log_stream(selected)
        return selected.index

    def _log_stream(self, stream: AudioStream) -> None:
        ch = "stereo" if stream.channels == 2 else "mono"
        logger.info(
            "Track #{}: {:<6}  {:<8}  {}kHz  {}",
            stream.index,
            stream.language or "?",
            stream.codec,
            stream.sample_rate // 1000,
            ch,
        )

    @staticmethod
    def _cache_key(video_path: str, stream_index: int, *, start: float | None = None, end: float | None = None) -> str:
        real = os.path.realpath(video_path)
        mtime = os.path.getmtime(real)
        raw = f"{real}|{mtime}|{stream_index}|{start}|{end}"
        return hashlib.md5(raw.encode()).hexdigest()


def _interactive_select(streams: list[AudioStream]) -> AudioStream:
    lines = [
        f"{s.index:>3}  {s.language or '?':<6}  {s.codec:<8}  {s.sample_rate // 1000:>3}kHz  {'stereo' if s.channels == 2 else 'mono'}"
        for s in streams
    ]

    fzf_path = shutil.which(get("fzf", "path", fallback="fzf"))
    if fzf_path:
        try:
            result = subprocess.run(
                [fzf_path, "--header", get("fzf", "header", fallback="Select audio track to extract:")],
                input="\n".join(lines),
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if result.returncode == 0 and result.stdout.strip():
                idx = int(result.stdout.strip().split()[0])
                return next(s for s in streams if s.index == idx)
        except (OSError, subprocess.SubprocessError):
            pass

    return _prompt_select(streams, lines)


def _prompt_select(
    streams: list[AudioStream], lines: list[str]
) -> AudioStream:
    print("\nMultiple audio tracks found. Select one:")
    for i, line in enumerate(lines):
        print(f"  [{i}] {line}")
    while True:
        try:
            choice = input(f"Enter 0-{len(streams) - 1}: ").strip()
            idx = int(choice)
            if 0 <= idx < len(streams):
                return streams[idx]
        except (ValueError, EOFError):
            pass
        print("Invalid selection, try again.")


def _track_ffmpeg_progress(
    proc: subprocess.Popen[str],
    duration: float,
    on_progress: ProgressCallback,
) -> None:
    for line in proc.stdout or []:
        line = line.strip()
        if line.startswith("out_time="):
            time_str = line[9:]
            current_time = _parse_ffmpeg_time(time_str)
            if duration > 0:
                on_progress(
                    ProgressEvent(
                        "ffmpeg",
                        current_time,
                        duration,
                        f"Extracting audio ({current_time:.1f}s / {duration:.1f}s)",
                    )
                )
        elif line == "progress=end":
            break


def _parse_ffmpeg_time(time_str: str) -> float:
    parts = time_str.strip().split(":")
    if len(parts) == 3:
        try:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        except ValueError:
            return 0.0
    return 0.0


_AUDIO_EXTENSIONS: frozenset | None = None


def _get_audio_extensions() -> frozenset:
    global _AUDIO_EXTENSIONS
    if _AUDIO_EXTENSIONS is None:
        exts = get_list("audio", "extensions")
        if exts:
            _AUDIO_EXTENSIONS = frozenset(exts)
        else:
            _AUDIO_EXTENSIONS = frozenset({
                ".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".opus", ".wma",
            })
    return _AUDIO_EXTENSIONS


def is_audio_only(path: str) -> bool:
    ext = Path(path).suffix.lower()
    if ext in _get_audio_extensions():
        return True
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                "-select_streams", "v",
                path,
            ],
            capture_output=True, text=True, check=True, encoding="utf-8",
        )
        data = json.loads(result.stdout)
        return len(data.get("streams", [])) == 0
    except Exception:
        return False
