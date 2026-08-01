from __future__ import annotations

from src.core.events import EventBus
from src.entity.context import PipelineContext
from src.entity.proof import ProofArgs
from src.service import audio as audio_service
from src.service.audio import StreamSelector


class AudioHandler:
    def __init__(
        self,
        args: ProofArgs,
        context: PipelineContext,
        bus: EventBus,
        track_selector: StreamSelector | None = None,
    ) -> None:
        self.args = args
        self.context = context
        self.bus = bus
        self.track_selector = track_selector

    def extract_audio(self) -> None:
        self.bus.log("probing audio streams...")
        streams = audio_service.probe_audio_streams(self.args.input_path)
        if len(streams) == 1:
            self.bus.log(f"1 audio track found, auto-selected index {streams[0]['index']}")
        elif len(streams) > 1:
            self.bus.log(f"{len(streams)} audio tracks found, prompting for selection")
        stream_index = audio_service.select_audio_stream(
            streams, prompt=self.track_selector
        )

        output_path = self.context.run_dir / f"{self.args.input_path.stem}.wav"
        self.bus.log(f"extracting audio track {stream_index} ...")
        audio_service.extract_audio(
            self.args.input_path,
            output_path,
            stream_index,
            on_progress=lambda pct: self.bus.set_progress("extract_audio", pct),
        )
        self.bus.set_progress("extract_audio", 100.0)
        self.bus.log("audio extraction complete")

        self.context.audio_path = output_path
        self.context.audio_track = stream_index
