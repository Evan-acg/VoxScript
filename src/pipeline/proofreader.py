from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from ..config import get
from ..progress import ProgressCallback, ProgressEvent, null_callback
from . import AudioExtractor, Options, Segment, resolve_output_path
from .aligner import SubtitleAligner
from .ass_formatter import format_ass
from .audio import is_audio_only
from .llm import LLMClient, ProofreadError
from .proof_report import (
    ExtraLineInfo,
    MissingLineInfo,
    ProofReport,
    ReportSummary,
    format_json_report,
    format_terminal_summary,
)
from .subtitle import SrtFormatter, Transcriber
from .subtitle_parser import SubtitleEvent, _format_srt_time, _parse_srt_time, parse_subtitle

_SUBTITLE_EXTS = {".srt", ".ssa", ".ass"}


def _hash_file(path: str, *, start: float | None = None, end: float | None = None) -> str:
    size = os.path.getsize(path)
    h = hashlib.md5()
    h.update(f"size:{size}|start:{start}|end:{end}".encode())
    with open(path, "rb") as f:
        chunk_size = 65536
        if size <= chunk_size * 4:
            h.update(f.read())
        else:
            h.update(f.read(chunk_size))
            mid = size // 2
            f.seek(mid - chunk_size // 2)
            h.update(f.read(chunk_size))
            f.seek(size - chunk_size)
            h.update(f.read(chunk_size))
    return h.hexdigest()


def _hash_subtitle(path: str) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _load_cache_db(cache_dir: Path) -> dict:
    cache_db_path = cache_dir / "cache.json"
    if cache_db_path.exists():
        return json.loads(cache_db_path.read_text(encoding="utf-8"))
    return {}


def _save_cache_db(cache_dir: Path, cache_db: dict) -> None:
    (cache_dir / "cache.json").write_text(
        json.dumps(cache_db, ensure_ascii=False, indent=2), encoding="utf-8"
    )


class Proofreader:
    def __init__(
        self,
        audio_extractor: AudioExtractor,
        transcriber: Transcriber,
        llm_client: LLMClient,
    ) -> None:
        self._audio_extractor = audio_extractor
        self._transcriber = transcriber
        self._llm_client = llm_client

    def run(
        self,
        video_path: str,
        subtitle_path: str,
        opts: Options,
        on_progress: ProgressCallback = null_callback,
    ) -> str:
        start_time = time.time()
        phases: dict[str, float] = {}
        cache_hits: list[str] = []

        t0 = time.time()
        on_progress(
            ProgressEvent("proofread", 0, 6, "Parsing input subtitle...")
        )
        doc = parse_subtitle(subtitle_path)
        phases["parse"] = time.time() - t0

        input_ext = Path(video_path).suffix.lower()
        if input_ext in _SUBTITLE_EXTS:
            return self._run_text_only(
                video_path, doc, opts, on_progress, start_time
            )

        audio_path = self._extract_audio(
            video_path, opts, cache_hits, on_progress
        )
        phases["extract"] = time.time() - t0
        t0 = time.time()

        on_progress(
            ProgressEvent("proofread", 1, 6, "Detecting audio language...")
        )
        from .aligner import SubtitleAligner as _Aligner

        try:
            language = _Aligner.detect_language(audio_path, opts.device)
        except Exception:
            language = opts.language or "en"
        phases["detect_lang"] = time.time() - t0
        t0 = time.time()

        on_progress(
            ProgressEvent("proofread", 2, 6, "Aligning subtitle to audio...")
        )
        aligner = SubtitleAligner(device=opts.device)
        cache_dir = Path.cwd() / ".vox_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        sub_hash = _hash_subtitle(subtitle_path)
        vid_hash = _hash_file(video_path, start=opts.start, end=opts.end)
        align_cache_key = f"{vid_hash}_{sub_hash[:12]}"
        align_cache_path = cache_dir / f"{align_cache_key}.alignment.json"

        aligned = None
        if not opts.force and align_cache_path.exists():
            try:
                data = json.loads(align_cache_path.read_text(encoding="utf-8"))
                from .aligner import AlignedSegment, AlignmentResult

                segments = [
                    AlignedSegment(**s) for s in data["segments"]
                ]
                aligned = AlignmentResult(
                    segments=segments,
                    language=data["language"],
                    avg_confidence=data["avg_confidence"],
                )
                cache_hits.append("对齐")
                on_progress(
                    ProgressEvent("alignment_cache", 1, 1, "Alignment cache hit")
                )
            except Exception:
                aligned = None

        if aligned is None:
            aligned = aligner.align(audio_path, doc.events, language, on_progress)
            align_cache_path.write_text(
                json.dumps(
                    {
                        "segments": [
                            {
                                "text": s.text,
                                "start": s.start,
                                "end": s.end,
                                "confidence": s.confidence,
                                "words": [
                                    {"text": w.text, "start": w.start, "end": w.end, "confidence": w.confidence}
                                    for w in s.words
                                ],
                            }
                            for s in aligned.segments
                        ],
                        "language": aligned.language,
                        "avg_confidence": aligned.avg_confidence,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        phases["align"] = time.time() - t0
        t0 = time.time()

        if aligned.avg_confidence >= 0.3:
            result = self._handle_same_language(
                doc,
                aligned,
                audio_path,
                opts,
                cache_hits,
                on_progress,
            )
        else:
            result = self._handle_translation(
                doc,
                video_path,
                audio_path,
                aligned,
                opts,
                cache_hits,
                on_progress,
            )
        phases["build"] = time.time() - t0

        report = result["report"]
        report.execution_info.phases = phases
        report.execution_info.cache_hits = cache_hits
        report.summary.elapsed = time.time() - start_time

        on_progress(ProgressEvent("proofread", 5, 6, "Saving report..."))
        self._save_report(report, Path(result["output_path"]))

        print("\n" + format_terminal_summary(report))
        return str(result["output_path"])

    def _extract_audio(
        self,
        video_path: str,
        opts: Options,
        cache_hits: list[str],
        on_progress: ProgressCallback,
    ) -> str:
        cache_dir = Path.cwd() / ".vox_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_db = _load_cache_db(cache_dir)

        input_hash = _hash_file(video_path, start=opts.start, end=opts.end)
        entry = cache_db.get(input_hash, {})
        cache_valid = not opts.force and bool(entry)

        if cache_valid and entry.get("audio") and Path(entry["audio"]).exists():
            cache_hits.append("音轨")
            on_progress(
                ProgressEvent("cache", 1, 1, "Audio cache hit")
            )
            return entry["audio"]

        if is_audio_only(video_path):
            on_progress(
                ProgressEvent("audio_check", 1, 1, "Audio-only file, skipping extraction")
            )
            cache_db[input_hash] = {
                "hash": input_hash,
                "path": str(Path(video_path).resolve()),
                "type": "audio",
                "audio": video_path,
            }
            _save_cache_db(cache_dir, cache_db)
            return video_path

        output_wav = str(cache_dir / f"{input_hash}.wav")
        self._audio_extractor.extract(
            video_path,
            output_wav,
            stream_index=opts.track_index,
            force=opts.force,
            start=opts.start,
            end=opts.end,
            on_progress=on_progress,
        )

        entry = cache_db.get(input_hash, {})
        cache_db[input_hash] = {
            "hash": input_hash,
            "path": str(Path(video_path).resolve()),
            "type": "video",
            "start": opts.start,
            "end": opts.end,
            "og_start": opts.og_start,
            "og_end": opts.og_end,
            "audio": output_wav,
        }
        _save_cache_db(cache_dir, cache_db)
        return output_wav

    def _handle_same_language(
        self,
        doc,
        aligned,
        audio_path: str,
        opts: Options,
        cache_hits: list[str],
        on_progress: ProgressCallback,
    ) -> dict:
        import whisperx

        on_progress(
            ProgressEvent("proofread", 3, 6, "Analyzing alignment...")
        )

        issues = SubtitleAligner.analyze_timeline(doc.events, aligned)

        audio = whisperx.load_audio(audio_path)
        sample_rate = get("audio", "sample_rate", fallback=16000)
        audio_dur = audio.shape[-1] / sample_rate

        gaps = SubtitleAligner.detect_gaps(audio_dur, aligned.segments)

        low_conf = SubtitleAligner.find_low_confidence_segments(aligned, 0.3)
        extra_lines: list[ExtraLineInfo] = []
        for idx_in_aligned, text, conf in low_conf:
            for ev in doc.events:
                if ev.text.strip() == text.strip():
                    extra_lines.append(
                        ExtraLineInfo(
                            line_index=ev.index,
                            text=ev.text[:80],
                            confidence=conf,
                        )
                    )
                    break

        missing_lines: list[MissingLineInfo] = [
            MissingLineInfo(g.start, g.end, g.duration, g.suggested_text)
            for g in gaps
        ]

        aligned_by_text: dict[str, float] = {}
        for s in aligned.segments:
            aligned_by_text[s.text.strip()] = s.confidence

        corrected_events: list[SubtitleEvent] = []
        aligned_seg_map: dict[str, tuple[float, float]] = {}
        for s in aligned.segments:
            aligned_seg_map[s.text.strip()] = (s.start, s.end)

        for i, ev in enumerate(doc.events):
            key = ev.text.strip()
            new_start, new_end = ev.start, ev.end
            if key in aligned_seg_map:
                new_start, new_end = aligned_seg_map[key]
            corrected_events.append(
                SubtitleEvent(
                    index=i + 1,
                    start=new_start,
                    end=new_end,
                    text=ev.text,
                )
            )

        video_name = Path(video_path).stem
        output_path = self._save_corrected_subtitle(corrected_events, video_name, opts)
        on_progress(
            ProgressEvent("proofread", 4, 6, f"Saved to {output_path}")
        )

        report = ProofReport(
            summary=ReportSummary(
                total_lines=len(doc.events),
                timeline_issues=len(issues),
                missing_lines=len(missing_lines),
                extra_lines=len(extra_lines),
                cache_hits=cache_hits,
            ),
            timeline_issues=issues,
            suspected_extra_lines=extra_lines,
            suspected_missing_lines=missing_lines,
        )

        return {
            "output_path": str(output_path),
            "report": report,
            "report_path": None,
        }

    def _handle_translation(
        self,
        doc,
        video_path: str,
        audio_path: str,
        aligned_initial,
        opts: Options,
        cache_hits: list[str],
        on_progress: ProgressCallback,
    ) -> dict:

        on_progress(
            ProgressEvent("proofread", 3, 6, "Translation mode: ASR transcription...")
        )

        original_text = "\n".join(
            f"{e.index}\n{_format_srt_time(e.start)} --> {_format_srt_time(e.end)}\n{e.text}"
            for e in doc.events
        )

        cache_dir = Path.cwd() / ".vox_cache"
        input_hash = _hash_file(video_path, start=opts.start, end=opts.end)
        cache_db = _load_cache_db(cache_dir)
        entry = cache_db.get(input_hash, {})
        cache_valid = not opts.force and bool(entry)

        ref_transcript: str | None = None
        if cache_valid and entry.get("whisper") and Path(entry["whisper"]).exists():
            ref_transcript = Path(entry["whisper"]).read_text(encoding="utf-8")
            cache_hits.append("转录")
            on_progress(
                ProgressEvent("transcription_cache", 1, 1, "Transcription cache hit")
            )

        if ref_transcript is None:
            on_progress(
                ProgressEvent("whisperx", 0, 0, "Transcribing audio...")
            )
            result = self._transcriber.transcribe(
                audio_path,
                model_name=opts.model_name,
                language=opts.language,
                device=opts.device,
                on_progress=on_progress,
            )

            if opts.start:
                result.segments = [
                    Segment(s.start + opts.start, s.end + opts.start, s.text)
                    for s in result.segments
                ]

            formatter = SrtFormatter()
            ref_transcript = formatter.format(result.segments)

            whisper_path = str(cache_dir / f"{input_hash}.whisper.srt")
            Path(whisper_path).write_text(ref_transcript, encoding="utf-8")

            entry = cache_db.get(input_hash, {})
            cache_db[input_hash] = {
                "hash": input_hash,
                "path": str(Path(video_path).resolve()),
                "type": "audio" if is_audio_only(video_path) else "video",
                "start": opts.start,
                "end": opts.end,
                "og_start": opts.og_start,
                "og_end": opts.og_end,
                "audio": audio_path,
                "whisper": whisper_path,
            }
            _save_cache_db(cache_dir, cache_db)

        translation_issues = None
        if opts.llm_check:
            on_progress(
                ProgressEvent("llm", 0, 0, "LLM translation check...")
            )
            try:
                llm_result = self._llm_client.proofread(
                    original_subtitle=original_text,
                    ref_transcript=ref_transcript,
                    subtitle_format=doc.format,
                    on_progress=on_progress,
                )
                corrected_segments = llm_result.get("corrected_segments", [])
            except ProofreadError:
                corrected_segments = []
        else:
            corrected_segments = []

        if corrected_segments:
            corrected_events = []
            for i, seg in enumerate(corrected_segments, start=1):
                corrected_events.append(
                    SubtitleEvent(
                        index=i,
                        start=_parse_srt_time(seg["start"]),
                        end=_parse_srt_time(seg["end"]),
                        text=seg["text"],
                    )
                )
        else:
            corrected_events = [
                SubtitleEvent(
                    index=i + 1,
                    start=ev.start,
                    end=ev.end,
                    text=ev.text,
                )
                for i, ev in enumerate(doc.events)
            ]

        video_name = Path(video_path).stem
        output_path = self._save_corrected_subtitle(corrected_events, video_name, opts)
        on_progress(
            ProgressEvent("proofread", 4, 6, f"Saved to {output_path}")
        )

        report = ProofReport(
            summary=ReportSummary(
                total_lines=len(doc.events),
                translation_issues=0,
                cache_hits=cache_hits,
            ),
            translation_issues=[],
        )

        return {
            "output_path": str(output_path),
            "report": report,
            "report_path": None,
        }

    def _run_text_only(
        self,
        ref_path: str,
        doc,
        opts: Options,
        on_progress: ProgressCallback,
        start_time: float,
    ) -> str:
        from .proof_report import ExecutionInfo

        on_progress(
            ProgressEvent("proofread", 1, 4, "Loading reference subtitle...")
        )
        ref_doc = parse_subtitle(ref_path)
        ref_transcript = "\n".join(
            f"{e.index}\n{_format_srt_time(e.start)} --> {_format_srt_time(e.end)}\n{e.text}"
            for e in ref_doc.events
        )

        original_text = "\n".join(
            f"{e.index}\n{_format_srt_time(e.start)} --> {_format_srt_time(e.end)}\n{e.text}"
            for e in doc.events
        )

        on_progress(
            ProgressEvent("llm", 0, 0, "LLM proofreading...")
        )
        llm_result = self._llm_client.proofread(
            original_subtitle=original_text,
            ref_transcript=ref_transcript,
            subtitle_format=doc.format,
            on_progress=on_progress,
        )

        corrected_segments = llm_result.get("corrected_segments", [])
        corrected_events = []
        for i, seg in enumerate(corrected_segments, start=1):
            corrected_events.append(
                SubtitleEvent(
                    index=i,
                    start=_parse_srt_time(seg["start"]),
                    end=_parse_srt_time(seg["end"]),
                    text=seg["text"],
                )
            )

        video_name = Path(ref_path).stem
        output_path = resolve_output_path(opts.output_dir, video_name, ".ass")

        from .subtitle_parser import SubtitleDocument

        corrected_doc = SubtitleDocument(format="ass", header="", events=corrected_events)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(format_ass(corrected_doc), encoding="utf-8")

        report = ProofReport(
            summary=ReportSummary(
                total_lines=len(doc.events),
                elapsed=time.time() - start_time,
            ),
            execution_info=ExecutionInfo(),
        )
        self._save_report(report, output_path)

        on_progress(
            ProgressEvent("proofread", 4, 4, f"Proofread subtitle saved to {output_path}")
        )
        return str(output_path)

    def _save_corrected_subtitle(
        self, events: list[SubtitleEvent], stem: str, opts: Options
    ) -> Path:
        output_path = resolve_output_path(opts.output_dir, stem, ".ass")

        from .subtitle_parser import SubtitleDocument

        doc = SubtitleDocument(format="ass", header="", events=events)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(format_ass(doc), encoding="utf-8")
        return output_path

    @staticmethod
    def _save_report(report: ProofReport, output_path: Path) -> str | None:
        try:
            report_path = output_path.with_suffix(".report.json")
            report_path.write_text(
                format_json_report(report), encoding="utf-8"
            )
            return str(report_path)
        except Exception:
            return None
