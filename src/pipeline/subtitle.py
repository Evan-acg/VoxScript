from __future__ import annotations

import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

import whisperx
from loguru import logger

from ..config import get, get_bool, get_float, get_int
from ..progress import ProgressCallback, ProgressEvent, null_callback
from . import Formatter, Segment, Transcriber, TranscriptionResult


class TranscriptionError(Exception):
    pass


def detect_language(audio_path: str, device: str) -> str:
    from ..config import get, get_bool

    import whisperx

    audio = whisperx.load_audio(audio_path)
    model = whisperx.load_model(
        "tiny",
        device=device,
        compute_type=get(
            "whisper",
            f"compute_type_{device}",
            fallback="float16" if device == "cuda" else "float32",
        ),
        language=None,
        local_files_only=get_bool("whisper", "local_files_only", fallback=False),
    )
    raw = model.transcribe(audio, batch_size=8)
    lang: str = raw.get("language", "en")
    return lang


class WhisperXTranscriber:
    def transcribe(
        self,
        audio_path: str,
        *,
        model_name: str,
        language: str | None,
        device: str,
        on_progress: ProgressCallback = null_callback,
    ) -> TranscriptionResult:
        from whisperx import load_audio

        audio = load_audio(audio_path)
        sample_rate = get_int("audio", "sample_rate", fallback=16000)
        audio_dur = audio.shape[-1] / sample_rate
        logger.info(
            "Audio loaded: {} samples @ {}Hz = {:.1f}s",
            audio.shape[-1], sample_rate, audio_dur,
        )

        model_dir_cfg = get("whisper", "model_dir")
        download_root = str(Path(__file__).resolve().parents[2] / model_dir_cfg) if model_dir_cfg else None

        on_progress(
            ProgressEvent("whisperx_load", 0, 0, f"Loading model '{model_name}'...")
        )
        try:
            model = whisperx.load_model(
                model_name,
                device=device,
                compute_type=get("whisper", f"compute_type_{device}", fallback="float16" if device == "cuda" else "float32"),
                language=language,
                download_root=download_root,
                local_files_only=get_bool("whisper", "local_files_only", fallback=False),
            )
        except Exception as e:
            raise TranscriptionError(f"Failed to load model: {e}") from e

        on_progress(
            ProgressEvent("whisperx_load", 1, 1, "Model loaded")
        )

        bridge = _WhisperProgressBridge(
            on_progress, "whisperx", total=audio_dur,
        )

        heartbeat = bridge.start_heartbeat()
        try:
            raw = model.transcribe(
                audio,
                batch_size=get_int("whisper", "batch_size", fallback=8),
                language=language,
                progress_callback=bridge.whisperx_callback(),
            )
        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}") from e
        finally:
            bridge.stop()
            heartbeat.join(timeout=get_int("progress", "heartbeat_timeout", fallback=2))

        detected_lang = raw.get("language", language or get("whisper", "fallback_language", fallback="en"))
        raw_segments = raw.get("segments", [])
        logger.info(
            "Transcribed: {} segments, lang={}", len(raw_segments), detected_lang,
        )
        on_progress(
            ProgressEvent("whisperx", audio_dur, audio_dur, "Transcription done")
        )

        on_progress(
            ProgressEvent(
                "whisperx_align_load", 0, 0, f"Loading align model (lang: {detected_lang})..."
            )
        )

        align_dir = str(Path(tempfile.gettempdir()) / "voxscript" / "align_models")
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=detected_lang, device=device,
                model_dir=align_dir, model_cache_only=get_bool("whisper", "align_cache_only", fallback=True),
            )
        except Exception:
            model_a, metadata = whisperx.load_align_model(
                language_code=detected_lang, device=device,
                model_dir=align_dir,
            )
        on_progress(
            ProgressEvent("whisperx_align_load", 1, 1, "Align model loaded")
        )

        if model_a is not None:
            align_bridge = _WhisperProgressBridge(
                on_progress, "whisperx_align",
            )
            align_hb = align_bridge.start_heartbeat()
            try:
                aligned = whisperx.align(
                    raw_segments,
                    model_a,
                    metadata,
                    audio,
                    device,
                    progress_callback=align_bridge.whisperx_callback(),
                )
                segments_raw = aligned["segments"]
                logger.info(
                    "Aligned: {} segments", len(segments_raw)
                )
            except Exception as e:
                raise TranscriptionError(
                    f"Alignment failed: {e}"
                ) from e
            finally:
                align_bridge.stop()
                align_hb.join(timeout=get_int("progress", "heartbeat_timeout", fallback=2))
        else:
            segments_raw = raw_segments

        segments = [
            Segment(start=s["start"], end=s["end"], text=s["text"].strip())
            for s in segments_raw
        ]

        last_end = segments[-1].end if segments else 0
        logger.info(
            "Transcribed %.1fs / %.1fs audio (%.0f%%)",
            last_end, audio_dur, last_end / audio_dur * 100,
        )

        on_progress(
            ProgressEvent("whisperx", 1, 1, f"Transcribed {len(segments)} segments")
        )

        return TranscriptionResult(segments=segments, language=detected_lang)


class SrtFormatter:
    extension: str = "srt"

    def format(self, segments: list[Segment]) -> str:
        lines: list[str] = []
        for i, seg in enumerate(segments, 1):
            lines.append(str(i))
            lines.append(
                f"{_format_time(seg.start)} --> {_format_time(seg.end)}"
            )
            lines.append(seg.text)
            lines.append("")
        return "\n".join(lines)


def _format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


class _WhisperProgressBridge:
    def __init__(
        self,
        on_progress: ProgressCallback,
        stage: str,
        total: float = 1.0,
    ) -> None:
        self._on_progress = on_progress
        self._stage = stage
        self._total = total
        self._actual = 0.0
        self._stop = threading.Event()

    def whisperx_callback(self) -> Callable[[float], None]:
        def callback(fraction: float) -> None:
            self._actual = fraction

        return callback

    def start_heartbeat(self) -> threading.Thread:
        interval = get_float("progress", "heartbeat_interval", fallback=0.5)

        def _heartbeat() -> None:
            while not self._stop.wait(interval):
                self._on_progress(
                    ProgressEvent(self._stage, self._actual * self._total, self._total, "")
                )

        t = threading.Thread(target=_heartbeat, daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self._stop.set()
