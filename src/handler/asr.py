from __future__ import annotations

import time
from pathlib import Path

from src.core.events import EventBus
from src.entity.context import PipelineContext
from src.entity.proof import ProofArgs
from src.service import asr as asr_service


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
        self.model = None

    def load_model(self) -> None:
        self.bus.log("loading whisperx model ...")
        start = time.perf_counter()
        self.model = asr_service.load_model(
            self.model_dir,
            self.model_name,
            on_log=lambda message: self.bus.log(message, level="WARNING"),
        )
        elapsed = time.perf_counter() - start
        self.bus.log(f"whisperx model loaded in {elapsed:.1f}s")

    def transcribe(self) -> None:
        if self.context.audio_path is None:
            raise RuntimeError("no audio path in context; run audio extraction first")
        if self.model is None:
            raise RuntimeError("whisperx model not loaded; call load_model first")

        self.bus.log("loading audio ...")
        self.bus.log("transcribing audio ...")
        segments = asr_service.transcribe(
            self.model,
            self.context.audio_path,
            language=self.language,
            on_progress=lambda pct: self.bus.set_progress("transcribe", pct),
        )

        output_path = self.context.run_dir / f"{self.args.input_path.stem}.srt"
        asr_service.write_srt(segments, output_path)

        self.context.transcript_path = output_path
        self.bus.log(f"transcription complete: {len(segments)} segments")
