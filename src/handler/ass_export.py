from __future__ import annotations

from src.core.events import EventBus
from src.entity.context import PipelineContext
from src.entity.proof import ProofArgs
from src.service.ass_export import export_ass_file


class AssExportHandler:
    def __init__(
        self,
        args: ProofArgs,
        context: PipelineContext,
        bus: EventBus,
    ) -> None:
        self.args = args
        self.context = context
        self.bus = bus

    def export(self) -> None:
        source_paths = (
            self.context.translated_paths
            or self.context.mapped_paths
        )
        if not source_paths:
            raise RuntimeError(
                "no translated or mapped subtitles in context; "
                "run map_timeline (or translate) first"
            )
        source = source_paths[-1]
        output_path = self.args.output_path
        if output_path is None:
            raise RuntimeError("no output path resolved for ASS export")
        export_ass_file(source, output_path)
        self.bus.log(
            f"ASS exported: {source.name} -> {output_path}"
        )
