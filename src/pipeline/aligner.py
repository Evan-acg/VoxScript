from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ..config import get, get_bool, get_int
from ..progress import ProgressCallback, ProgressEvent, null_callback
from .subtitle_parser import SubtitleEvent


@dataclass(frozen=True)
class WordTiming:
    text: str
    start: float
    end: float
    confidence: float


@dataclass(frozen=True)
class AlignedSegment:
    text: str
    start: float
    end: float
    confidence: float
    words: list[WordTiming] = field(default_factory=list)


@dataclass(frozen=True)
class AlignmentResult:
    segments: list[AlignedSegment]
    language: str
    avg_confidence: float


@dataclass(frozen=True)
class GapRegion:
    start: float
    end: float
    duration: float
    suggested_text: str | None = None


class AlignmentError(Exception):
    pass


def _clean_text(text: str) -> str:
    text = re.sub(r"\{[^}]*\}", "", text)
    text = text.replace("\\N", " ").replace("\\n", " ").replace("\n", " ")
    return text.strip()


class SubtitleAligner:
    def __init__(self, device: str = "cuda") -> None:
        self._device = device

    @staticmethod
    def detect_language(audio_path: str, device: str) -> str:
        from loguru import logger

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
        logger.info("Detected audio language: {}", lang)
        return lang

    def align(
        self,
        audio_path_or_array,
        events: list[SubtitleEvent],
        language: str,
        on_progress: ProgressCallback = null_callback,
    ) -> AlignmentResult:
        import whisperx
        from loguru import logger

        if isinstance(audio_path_or_array, str):
            audio = whisperx.load_audio(audio_path_or_array)
        else:
            audio = audio_path_or_array

        on_progress(
            ProgressEvent("align_load", 0, 0, f"Loading align model ({language})...")
        )
        align_dir = str(Path(tempfile.gettempdir()) / "voxscript" / "align_models")
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=language,
                device=self._device,
                model_dir=align_dir,
                model_cache_only=get_bool("whisper", "align_cache_only", fallback=True),
            )
        except Exception:
            model_a, metadata = whisperx.load_align_model(
                language_code=language,
                device=self._device,
                model_dir=align_dir,
            )
        on_progress(ProgressEvent("align_load", 1, 1, "Align model loaded"))

        segments = [
            {"text": _clean_text(e.text), "start": e.start, "end": e.end}
            for e in events
        ]

        on_progress(ProgressEvent("align", 0, 0, "Aligning subtitle to audio..."))
        try:
            aligned = whisperx.align(segments, model_a, metadata, audio, self._device)
        except Exception as e:
            raise AlignmentError(f"Alignment failed: {e}") from e

        raw_segments = aligned.get("segments", [])
        result_segments: list[AlignedSegment] = []
        for s in raw_segments:
            words = [
                WordTiming(
                    text=w.get("text", ""),
                    start=w.get("start", 0.0),
                    end=w.get("end", 0.0),
                    confidence=w.get("score", 0.0),
                )
                for w in s.get("words", [])
            ]
            conf = s.get("confidence") or s.get("score", 0.0) or 0.0
            result_segments.append(
                AlignedSegment(
                    text=s.get("text", ""),
                    start=s.get("start", 0.0),
                    end=s.get("end", 0.0),
                    confidence=float(conf),
                    words=words,
                )
            )

        avg_conf = 0.0
        if result_segments:
            avg_conf = sum(s.confidence for s in result_segments) / len(result_segments)

        logger.info(
            "Aligned {} segments, avg confidence={:.2f}",
            len(result_segments),
            avg_conf,
        )
        on_progress(
            ProgressEvent(
                "align", 1, 1, f"Aligned {len(result_segments)} segments"
            )
        )

        return AlignmentResult(
            segments=result_segments,
            language=language,
            avg_confidence=avg_conf,
        )

    @staticmethod
    def analyze_timeline(
        events: list[SubtitleEvent],
        aligned: AlignmentResult,
        threshold: float = 0.3,
    ) -> list:
        from .proof_report import TimelineIssue as TI

        issues: list = []
        aligned_by_text: dict[str, AlignedSegment] = {}
        for seg in aligned.segments:
            key = _clean_text(seg.text)
            if key:
                aligned_by_text[key] = seg

        for ev in events:
            key = _clean_text(ev.text)
            a = aligned_by_text.get(key)
            if a is None:
                continue

            offset_s = a.start - ev.start
            offset_e = a.end - ev.end
            max_offset = max(abs(offset_s), abs(offset_e))

            if max_offset <= threshold:
                continue

            if max_offset > 1.0:
                severity = "error"
            elif max_offset > 0.5:
                severity = "warning"
            else:
                severity = "info"

            issues.append(
                TI(
                    line_index=ev.index,
                    text=ev.text[:80],
                    original_start=ev.start,
                    aligned_start=a.start,
                    original_end=ev.end,
                    aligned_end=a.end,
                    offset_start=offset_s,
                    offset_end=offset_e,
                    severity=severity,
                )
            )

        return issues

    @staticmethod
    def detect_gaps(
        audio_duration: float,
        aligned_segments: list[AlignedSegment],
        min_gap: float = 2.0,
    ) -> list[GapRegion]:
        covered = [(s.start, s.end) for s in aligned_segments]
        covered.sort(key=lambda x: x[0])

        gaps: list[GapRegion] = []
        cursor = 0.0
        for start, end in covered:
            if start > cursor and (start - cursor) >= min_gap:
                gaps.append(GapRegion(cursor, start, start - cursor))
            cursor = max(cursor, end)

        if audio_duration - cursor >= min_gap:
            gaps.append(
                GapRegion(cursor, audio_duration, audio_duration - cursor)
            )

        return gaps

    @staticmethod
    def find_low_confidence_segments(
        aligned: AlignmentResult, threshold: float = 0.3
    ) -> list[tuple[int, str, float]]:
        results: list[tuple[int, str, float]] = []
        for i, seg in enumerate(aligned.segments):
            if seg.confidence < threshold:
                results.append((i, seg.text, seg.confidence))
        return results
