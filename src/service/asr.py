from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

from faster_whisper import WhisperModel
from whisperx import load_audio

_SAMPLE_RATE = 16000
_CHUNK_SIZE = 30.0


def load_model(
    model_dir: Path,
    model_name: str,
    on_log: Callable[[str], None] | None = None,
):
    try:
        return WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            download_root=str(model_dir),
            local_files_only=True,
        )
    except (ValueError, OSError, RuntimeError):
        if on_log is not None:
            on_log("whisper model not in local cache, downloading ...")
        return WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            download_root=str(model_dir),
            local_files_only=False,
        )


def transcribe(
    model,
    audio_path: Path,
    language: str | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> list[dict]:
    audio = load_audio(str(audio_path))
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
        segments, info = model.transcribe(
            slice_audio,
            language=chunk_language,
            word_timestamps=True,
            vad_filter=True,
            beam_size=5,
        )
        if detected_language is None and info.language:
            detected_language = info.language

        for segment in segments:
            seg_start = float(segment.start)
            seg_end = float(segment.end)
            words = list(segment.words or [])
            if words:
                seg_start = float(words[0].start)
                seg_end = float(words[-1].end)
            all_segments.append(
                {
                    "start": round(seg_start + start, 3),
                    "end": round(seg_end + start, 3),
                    "text": str(segment.text).strip(),
                }
            )

        if on_progress is not None:
            on_progress((index + 1) / chunk_count * 100)

    return all_segments
