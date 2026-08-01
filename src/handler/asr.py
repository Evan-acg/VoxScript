from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path

import click
import whisperx
from huggingface_hub.errors import LocalEntryNotFoundError

from src.core.events import EventBus
from src.entity.context import PipelineContext
from src.entity.proof import ProofArgs

_SAMPLE_RATE = 16000
_CHUNK_SIZE = 30.0


class AsrHandler:
    def __init__(
        self,
        args: ProofArgs,
        context: PipelineContext,
        bus: EventBus,
        model_dir: Path,
        model_name: str,
        language: str | None = None,
    ) -> None:
        self.args = args
        self.context = context
        self.bus = bus
        self.model_dir = model_dir
        self.model_name = model_name
        self.language = language

    def transcribe(self) -> PipelineContext:
        if self.context.audio_path is None:
            raise RuntimeError("no audio path in context; run audio extraction first")

        self.bus.log("loading whisperx model ...")
        start = time.perf_counter()
        model = self._load_model()
        elapsed = time.perf_counter() - start
        self.bus.log(f"whisperx model loaded in {elapsed:.1f}s")

        self.bus.log("loading audio ...")
        audio = whisperx.load_audio(str(self.context.audio_path))

        self.bus.log("transcribing audio ...")
        segments = self._transcribe_chunks(model, audio)

        work_dir = Path(tempfile.mkdtemp(prefix="voxscript_"))
        output_path = work_dir / f"{self.args.input_path.stem}.srt"
        self._write_srt(segments, output_path)

        self.context.transcript_path = output_path
        self.bus.log(f"transcription complete: {len(segments)} segments")
        return self.context

    def _load_model(self):
        common = dict(
            device="cpu",
            compute_type="int8",
            download_root=str(self.model_dir),
            vad_method="silero",
        )
        try:
            return whisperx.load_model(self.model_name, local_files_only=True, **common)
        except LocalEntryNotFoundError:
            self.bus.log(
                "whisperx model not in local cache, downloading ...",
                level="WARNING",
            )
            try:
                return whisperx.load_model(self.model_name, local_files_only=False, **common)
            except Exception as exc:
                raise click.ClickException(
                    "failed to download whisperx model; if the network is slow, "
                    "set HF_ENDPOINT=https://hf-mirror.com in midterm.bat and retry"
                ) from exc

    def _transcribe_chunks(self, model, audio) -> list[dict]:
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

            language = self.language or detected_language
            result = model.transcribe(
                slice_audio,
                language=language,
                print_progress=False,
            )
            if detected_language is None and result.get("language"):
                detected_language = result["language"]

            for segment in result.get("segments", []):
                segment["start"] = round(float(segment["start"]) + start, 3)
                segment["end"] = round(float(segment["end"]) + start, 3)
                all_segments.append(segment)

            self.bus.set_progress("transcribe", (index + 1) / chunk_count * 100)

        return all_segments

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
