from __future__ import annotations

import tempfile
from pathlib import Path

from ..config import get
from ..progress import ProgressCallback, ProgressEvent, null_callback
from . import AudioExtractor, Options
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

    def run(
        self,
        video_path: str,
        subtitle_path: str,
        opts: Options,
        on_progress: ProgressCallback = null_callback,
    ) -> str:
        from pathlib import Path as _Path

        video_name = _Path(video_path).stem

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

        if is_audio_only(video_path):
            audio_path = video_path
            on_progress(
                ProgressEvent("proofread", 1, 4, "Audio-only file, skipping extraction")
            )
        else:
            tmpdir = tempfile.mkdtemp(prefix=get("logging", "temp_prefix", fallback="voxscript_"))
            output_wav = _Path(tmpdir, f"{video_name}.wav")
            self._audio_extractor.extract(
                video_path,
                str(output_wav),
                stream_index=opts.track_index,
                on_progress=on_progress,
            )
            audio_path = str(output_wav)

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

        formatter = SrtFormatter()
        ref_transcript = formatter.format(result.segments)

        on_progress(
            ProgressEvent("proofread", 3, 4, "LLM proofreading...")
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

        output_path = _Path(opts.output_dir, f"{video_name}.ass")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ass_content = format_ass(corrected_doc)
        output_path.write_text(ass_content, encoding="utf-8")

        on_progress(
            ProgressEvent("proofread", 4, 4, f"Proofread subtitle saved to {output_path}")
        )

        return str(output_path)
