from __future__ import annotations

from src.core.events import EventBus
from src.entity.context import PipelineContext
from src.entity.proof import ProofArgs
from src.service.subtitle import parse_subtitle_file, write_normalized_json


class SubtitleHandler:
    def __init__(
        self,
        args: ProofArgs,
        context: PipelineContext,
        bus: EventBus,
    ) -> None:
        self.args = args
        self.context = context
        self.bus = bus

    def normalize(self) -> None:
        if self.context.transcript_path is None:
            raise RuntimeError("no transcript path in context; run transcribe first")

        user_subtitles = []
        normalized_paths = []
        for path in self.args.subtitle_paths:
            subtitle_format = path.suffix.lstrip(".").lower()
            self.bus.log(f"parsing {path.name} ({subtitle_format}) ...")
            parsed = parse_subtitle_file(path)
            self.bus.log(f"{path.name}: {len(parsed.segments)} cues parsed")
            user_subtitles.append(parsed)

            output_path = self.context.run_dir / f"{path.stem}.normalized.json"
            write_normalized_json(parsed, output_path)
            normalized_paths.append(output_path)
            self.bus.log(f"normalized subtitle written: {output_path}")

        self.context.user_subtitles = user_subtitles
        self.context.normalized_paths = normalized_paths

        transcript_path = self.context.transcript_path
        self.bus.log(f"parsing transcript {transcript_path.name} ...")
        transcript = parse_subtitle_file(transcript_path, label="transcript")
        self.bus.log(f"transcript: {len(transcript.segments)} cues parsed")

        self.context.transcript_segments = transcript.segments
