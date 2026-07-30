from __future__ import annotations

from pathlib import Path

from ..config import get
from ..progress import ProgressCallback, ProgressEvent, null_callback
from . import AudioExtractor, Options, Segment
from .ass_formatter import format_ass
from .audio import is_audio_only
from .llm import LLMClient, ProofreadError
from .subtitle import SrtFormatter, Transcriber
from .subtitle_parser import parse_subtitle, _format_srt_time, _parse_srt_time


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

    @staticmethod
    def _hash_file(path: str, *, start: float | None = None, end: float | None = None) -> str:
        import hashlib
        import os
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

    def run(
        self,
        video_path: str,
        subtitle_path: str,
        opts: Options,
        on_progress: ProgressCallback = null_callback,
    ) -> str:
        from pathlib import Path as _Path

        video_name = _Path(video_path).stem
        output_path = _Path(opts.output_dir, f"{video_name}.ass")

        on_progress(
            ProgressEvent("proofread", 0, 4, "Parsing input subtitle...")
        )
        doc = parse_subtitle(subtitle_path)
        original_text = "\n".join(
            f"{e.index}\n{_format_srt_time(e.start)} --> {_format_srt_time(e.end)}\n{e.text}"
            for e in doc.events
        )
        on_progress(
            ProgressEvent("proofread", 1, 4, f"Parsed {len(doc.events)} subtitle events")
        )

        _SUBTITLE_EXTS = {".srt", ".ssa", ".ass"}
        input_ext = _Path(video_path).suffix.lower()

        if input_ext in _SUBTITLE_EXTS:
            on_progress(
                ProgressEvent("proofread", 1, 4, "Loading reference subtitle...")
            )
            ref_doc = parse_subtitle(video_path)
            ref_transcript = "\n".join(
                f"{e.index}\n{_format_srt_time(e.start)} --> {_format_srt_time(e.end)}\n{e.text}"
                for e in ref_doc.events
            )
            on_progress(
                ProgressEvent("proofread", 2, 4, f"Loaded {len(ref_doc.events)} reference segments")
            )
        else:
            import json as _json

            cache_dir = _Path.cwd() / ".vox_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_db_path = cache_dir / "cache.json"

            cache_db: dict = {}
            if cache_db_path.exists():
                cache_db = _json.loads(cache_db_path.read_text(encoding="utf-8"))

            input_hash = self._hash_file(video_path, start=opts.start, end=opts.end)
            input_type = "audio" if is_audio_only(video_path) else "video"
            entry = cache_db.get(input_hash, {})
            cache_valid = (
                not opts.force and bool(entry)
            )

            # --- Audio ---
            audio_path: str | None = None
            if cache_valid and entry.get("audio") and _Path(entry["audio"]).exists():
                audio_path = entry["audio"]
                on_progress(
                    ProgressEvent("proofread", 1, 4, "Audio cache hit")
                )

            if audio_path is None:
                if is_audio_only(video_path):
                    audio_path = video_path
                    on_progress(
                        ProgressEvent("proofread", 1, 4, "Audio-only file, skipping extraction")
                    )
                else:
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
                    audio_path = output_wav

            # --- Transcription ---
            ref_transcript: str | None = None
            whisper_path: str = ""
            if cache_valid and entry.get("whisper") and _Path(entry["whisper"]).exists():
                ref_transcript = _Path(entry["whisper"]).read_text(encoding="utf-8")
                whisper_path = entry["whisper"]
                on_progress(
                    ProgressEvent("proofread", 2, 4, "Transcription cache hit")
                )

            if ref_transcript is None:
                on_progress(
                    ProgressEvent("proofread", 2, 4, "Transcribing audio...")
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
                _Path(whisper_path).write_text(ref_transcript, encoding="utf-8")

            # --- Update cache ---
            cached_entry = cache_db.get(input_hash, {})
            cached_path = str(_Path(video_path).resolve())
            if (
                cached_entry.get("path") != cached_path
                or cached_entry.get("audio") != audio_path
                or cached_entry.get("whisper") != whisper_path
                or cached_entry.get("start") != opts.start
                or cached_entry.get("end") != opts.end
            ):
                cache_db[input_hash] = {
                    "hash": input_hash,
                    "path": str(_Path(video_path).resolve()),
                    "type": input_type,
                    "start": opts.start,
                    "end": opts.end,
                    "og_start": opts.og_start,
                    "og_end": opts.og_end,
                    "audio": audio_path,
                    "whisper": whisper_path,
                }
                cache_db_path.write_text(
                    _json.dumps(cache_db, ensure_ascii=False, indent=2), encoding="utf-8"
                )

            on_progress(
                ProgressEvent("proofread", 2, 4, "Reference loaded")
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
            from .subtitle_parser import SubtitleEvent

            corrected_events.append(
                SubtitleEvent(
                    index=i,
                    start=_parse_srt_time(seg["start"]),
                    end=_parse_srt_time(seg["end"]),
                    text=seg["text"],
                )
            )

        from .subtitle_parser import SubtitleDocument

        corrected_doc = SubtitleDocument(
            format="ass",
            header="",
            events=corrected_events,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ass_content = format_ass(corrected_doc)
        output_path.write_text(ass_content, encoding="utf-8")

        on_progress(
            ProgressEvent("proofread", 4, 4, f"Proofread subtitle saved to {output_path}")
        )

        return str(output_path)
