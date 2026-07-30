from __future__ import annotations

import logging
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

import whisperx

from ..config import get, get_bool, get_float, get_int, get_section
from ..progress import ProgressCallback, ProgressEvent, null_callback
from . import Formatter, Segment, Transcriber, TranscriptionResult

logger = logging.getLogger("voxscript")


class TranscriptionError(Exception):
    pass


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
            "Audio loaded: %d samples @ %dHz = %.1fs",
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
            on_progress, "whisperx", audio_dur, model_name, device,
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
            "Transcribed: %d segments, lang=%s", len(raw_segments), detected_lang,
        )
        on_progress(
            ProgressEvent("whisperx", 1, 1, "Transcription done")
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
                on_progress, "whisperx_align", audio_dur, model_name, device,
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
                    "Aligned: %d segments", len(segments_raw)
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


def _load_speed_factors() -> dict[tuple[str, str], float]:
    section = get_section("whisper", "speed_factors")
    result: dict[tuple[str, str], float] = {}
    for key, val in section.items():
        parts = key.split(",")
        if len(parts) == 2:
            try:
                result[(parts[0].strip(), parts[1].strip())] = float(val)
            except (ValueError, TypeError):
                pass
    return result


class _WhisperProgressBridge:
    def __init__(
        self,
        on_progress: ProgressCallback,
        stage: str,
        audio_dur: float,
        model_name: str,
        device: str,
    ) -> None:
        self._on_progress = on_progress
        self._stage = stage
        self._actual = 0.0
        self._start = time.time()
        factors = _load_speed_factors()
        factor = factors.get((model_name, device), 1)
        min_est = get_int("whisper", "min_estimate_seconds", fallback=10)
        self._estimate = max(audio_dur / factor, min_est)
        self._stop = threading.Event()

    def whisperx_callback(self) -> Callable[[float], None]:
        def callback(fraction: float) -> None:
            self._actual = fraction

        return callback

    def start_heartbeat(self) -> threading.Thread:
        interval = get_float("progress", "heartbeat_interval", fallback=0.5)
        max_frac = get_float("progress", "max_fraction", fallback=0.95)

        def _heartbeat() -> None:
            while not self._stop.wait(interval):
                elapsed = time.time() - self._start
                fraction = max(self._actual, min(elapsed / self._estimate, max_frac))
                self._on_progress(
                    ProgressEvent(self._stage, fraction, 1.0, "Transcribing...")
                )

        t = threading.Thread(target=_heartbeat, daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self._stop.set()
