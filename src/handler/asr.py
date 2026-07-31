from __future__ import annotations

import tempfile
from pathlib import Path

import whisperx

from src.core.events import EventBus
from src.entity.context import PipelineContext
from src.entity.proof import ProofArgs


class AsrHandler:
    def __init__(
        self,
        args: ProofArgs,
        context: PipelineContext,
        bus: EventBus,
        model_dir: Path,
        language: str | None = None,
    ) -> None:
        self.args = args
        self.context = context
        self.bus = bus
        self.model_dir = model_dir
        self.language = language

    def transcribe(self) -> PipelineContext:
        if self.context.audio_path is None:
            raise RuntimeError("no audio path in context; run audio extraction first")

        self.bus.log("loading whisperx model ...")
        model = whisperx.load_model(
            "tiny",
            device="cpu",
            compute_type="int8",
            download_root=str(self.model_dir),
            vad_method="silero",
        )

        self.bus.log("transcribing audio ...")
        result = model.transcribe(
            str(self.context.audio_path),
            language=self.language,
        )
        segments = result.get("segments", [])

        work_dir = Path(tempfile.mkdtemp(prefix="voxscript_"))
        output_path = work_dir / f"{self.args.input_path.stem}.srt"
        self._write_srt(segments, output_path)

        self.context.transcript_path = output_path
        self.bus.log(f"transcription complete: {len(segments)} segments")
        return self.context

    @staticmethod
    def _write_srt(segments: list[dict], output_path: Path) -> None:
        lines: list[str] = []
        for index, segment in enumerate(segments, start=1):
            start = _format_timestamp(float(segment["start"]))
            end = _format_timestamp(float(segment["end"]))
            text = str(segment["text"]).strip()
            lines.append(str(index))
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")
        output_path.write_text("\n".join(lines), encoding="utf-8")


def _format_timestamp(seconds: float) -> str:
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
