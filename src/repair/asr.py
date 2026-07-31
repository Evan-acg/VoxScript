from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..config import get, get_bool, get_int
from ..progress import ProgressCallback, ProgressEvent, null_callback
from .models import AsrSegment


class AsrError(Exception):
    pass


class WhisperXASR:
    def __init__(self) -> None:
        self._models: dict[tuple[str, str, str | None, str], Any] = {}
        self._align_models: dict[tuple[str, str], tuple[Any, Any]] = {}
        self.last_language: str | None = None

    def transcribe(
        self,
        audio_path: str,
        *,
        offset: float = 0.0,
        chunk_id: int = 0,
        model_name: str,
        language: str | None,
        device: str,
        vad_method: str = "silero",
        batch_size: int | None = None,
        on_progress: ProgressCallback = null_callback,
    ) -> list[AsrSegment]:
        try:
            import whisperx

            self._ensure_ffmpeg_path()
            audio = whisperx.load_audio(audio_path)
            sample_rate = get_int("audio", "sample_rate", fallback=16000)
            duration = audio.shape[-1] / sample_rate
            model = self._get_model(
                whisperx,
                model_name,
                language,
                device,
                vad_method,
                on_progress,
            )
            raw = model.transcribe(
                audio,
                batch_size=batch_size or get_int("whisper", "batch_size", fallback=8),
                language=language,
            )
            detected_language = raw.get("language") or language or get(
                "whisper",
                "fallback_language",
                fallback="en",
            )
            self.last_language = str(detected_language)
            segments = self._align(
                whisperx,
                raw.get("segments", []),
                audio,
                detected_language,
                device,
                on_progress,
            )
        except Exception as error:
            if isinstance(error, AsrError):
                raise
            raise AsrError(f"WhisperX transcription failed: {error}") from error

        result: list[AsrSegment] = []
        for index, segment in enumerate(segments):
            try:
                start = float(segment["start"]) + offset
                end = float(segment["end"]) + offset
                text = str(segment.get("text", "")).strip()
            except (KeyError, TypeError, ValueError):
                continue
            if end <= start or not text:
                continue
            result.append(
                AsrSegment(
                    id=index,
                    start=max(0.0, start),
                    end=max(0.0, end),
                    text=text,
                    chunk_id=chunk_id,
                )
            )

        on_progress(
            ProgressEvent(
                "asr",
                duration,
                duration,
                f"ASR produced {len(result)} segments",
            )
        )
        return result

    @staticmethod
    def _ensure_ffmpeg_path() -> None:
        configured = get("ffmpeg", "path", fallback="ffmpeg")
        configured_path = Path(configured)
        if configured_path.is_file():
            parent = str(configured_path.parent.resolve())
            current = os.environ.get("PATH", "").split(os.pathsep)
            if parent not in current:
                os.environ["PATH"] = os.pathsep.join([parent, *current])
            return
        if shutil.which("ffmpeg") is None:
            return

    def _get_model(
        self,
        whisperx: Any,
        model_name: str,
        language: str | None,
        device: str,
        vad_method: str,
        on_progress: ProgressCallback,
    ) -> Any:
        key = (model_name, device, language, vad_method)
        if key in self._models:
            return self._models[key]
        on_progress(ProgressEvent("asr", 0, 0, f"Loading WhisperX model '{model_name}'"))
        model_dir = get("whisper", "model_dir")
        download_root = str(Path(__file__).resolve().parents[2] / model_dir) if model_dir else None
        try:
            model = whisperx.load_model(
                model_name,
                device=device,
                compute_type=get(
                    "whisper",
                    f"compute_type_{device}",
                    fallback="float16" if device == "cuda" else "float32",
                ),
                language=language,
                vad_method=vad_method,
                download_root=download_root,
                local_files_only=get_bool("whisper", "local_files_only", fallback=False),
            )
        except Exception as error:
            raise AsrError(f"failed to load WhisperX model: {error}") from error
        self._models[key] = model
        on_progress(ProgressEvent("asr", 1, 1, "WhisperX model loaded"))
        return model

    def _align(
        self,
        whisperx: Any,
        segments: list[dict[str, Any]],
        audio: Any,
        language: str,
        device: str,
        on_progress: ProgressCallback,
    ) -> list[dict[str, Any]]:
        key = (language, device)
        if key not in self._align_models:
            align_dir = str(Path(tempfile.gettempdir()) / "voxscript" / "align_models")
            try:
                self._align_models[key] = whisperx.load_align_model(
                    language_code=language,
                    device=device,
                    model_dir=align_dir,
                    model_cache_only=get_bool("whisper", "align_cache_only", fallback=True),
                )
            except Exception:
                try:
                    self._align_models[key] = whisperx.load_align_model(
                        language_code=language,
                        device=device,
                        model_dir=align_dir,
                    )
                except Exception as error:
                    on_progress(
                        ProgressEvent(
                            "asr",
                            0,
                            0,
                            f"Alignment unavailable for {language}; using ASR segments",
                        )
                    )
                    self._align_models[key] = (None, None)
        model_a, metadata = self._align_models[key]
        if model_a is None:
            return segments
        on_progress(ProgressEvent("asr", 0, 0, f"Aligning {language} segments"))
        try:
            return whisperx.align(
                segments,
                model_a,
                metadata,
                audio,
                device,
            )["segments"]
        except Exception:
            on_progress(
                ProgressEvent(
                    "asr",
                    0,
                    0,
                    f"Alignment failed for {language}; using ASR segments",
                )
            )
            return segments
