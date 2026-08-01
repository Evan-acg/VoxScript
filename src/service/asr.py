from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import whisperx
from huggingface_hub.errors import LocalEntryNotFoundError

_SAMPLE_RATE = 16000
_CHUNK_SIZE = 30.0


def load_model(
    model_dir: Path,
    model_name: str,
    on_log: Callable[[str], None] | None = None,
):
    kwargs = dict(
        device="cpu",
        compute_type="int8",
        download_root=str(model_dir),
        vad_method="silero",
    )
    try:
        return whisperx.load_model(model_name, local_files_only=True, **kwargs)
    except LocalEntryNotFoundError:
        if on_log is not None:
            on_log("whisperx model not in local cache, downloading ...")
        return _download_model(model_name, kwargs)


def transcribe(
    model,
    audio_path: Path,
    language: str | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> list[dict]:
    audio = whisperx.load_audio(str(audio_path))
    return _transcribe_chunks(model, audio, language, on_progress)


def write_srt(segments: list[dict], output_path: Path) -> None:
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        start = format_timestamp(float(segment["start"]))
        end = format_timestamp(float(segment["end"]))
        text = str(segment["text"]).strip()
        lines.append(str(index))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int(seconds % 3600 // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:
        millis = 0
        secs += 1
        if secs == 60:
            secs = 0
            minutes += 1
            if minutes == 60:
                minutes = 0
                hours += 1
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _download_model(model_name: str, kwargs: dict):
    try:
        return whisperx.load_model(model_name, local_files_only=False, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"failed to download whisperx model: {exc}; if the network is slow, "
            "set HF_ENDPOINT=https://hf-mirror.com in midterm.bat and retry"
        ) from exc


def _transcribe_chunks(
    model,
    audio,
    language: str | None,
    on_progress: Callable[[float], None] | None,
) -> list[dict]:
    duration = len(audio) / _SAMPLE_RATE
    if duration <= 0:
        return []

    chunk_count = math.ceil(duration / _CHUNK_SIZE)
    detected_language: str | None = None
    all_segments: list[dict] = []

    for index in range(chunk_count):
        start = index * _CHUNK_SIZE
        end = min((index + 1) * _CHUNK_SIZE, duration)
        slice_audio = audio[
            int(start * _SAMPLE_RATE) : int(end * _SAMPLE_RATE)
        ]
        if len(slice_audio) == 0:
            continue

        chunk_language = language or detected_language
        result = model.transcribe(
            slice_audio,
            language=chunk_language,
            print_progress=False,
        )
        if detected_language is None and result.get("language"):
            detected_language = result["language"]

        for segment in result.get("segments", []):
            segment["start"] = round(float(segment["start"]) + start, 3)
            segment["end"] = round(float(segment["end"]) + start, 3)
            all_segments.append(segment)

        if on_progress is not None:
            on_progress((index + 1) / chunk_count * 100)

    return all_segments
