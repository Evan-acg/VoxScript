from __future__ import annotations

from src.core.events import EventBus
from src.entity.context import PipelineContext
from src.entity.proof import ProofArgs
from src.service.split import split_transcript_file


class SplitHandler:
    def __init__(
        self,
        args: ProofArgs,
        context: PipelineContext,
        bus: EventBus,
        language: str | None = None,
    ) -> None:
        self.args = args
        self.context = context
        self.bus = bus
        self.language = language

    def split(self) -> None:
        if self.context.transcript_normalized_path is None:
            raise RuntimeError(
                "no normalized transcript in context; run normalize_subtitles first"
            )

        input_path = self.context.transcript_normalized_path
        self.bus.log(f"splitting transcript {input_path.name} ...")
        output_path = (
            self.context.run_dir / f"{self.args.input_path.stem}.transcript.split.json"
        )
        split = split_transcript_file(
            input_path,
            output_path,
            language=self.language,
            on_log=lambda message: self.bus.log(message, level="WARNING"),
        )
        self.context.split_json_path = output_path
        self.bus.log(f"transcript split: {len(split.dialogue)} cues")
