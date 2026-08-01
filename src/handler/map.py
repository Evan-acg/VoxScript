from __future__ import annotations

from src.core.events import EventBus
from src.entity.context import PipelineContext
from src.entity.proof import ProofArgs
from src.entity.subtitle import NormalizedSubtitle
from src.service.map import map_transcript


class MapHandler:
    def __init__(
        self,
        args: ProofArgs,
        context: PipelineContext,
        bus: EventBus,
        language: str | None = None,
        english_style: str = "Eng",
    ) -> None:
        self.args = args
        self.context = context
        self.bus = bus
        self.language = language
        self.english_style = english_style

    def map_timelines(self) -> None:
        if self.context.split_json_path is None:
            raise RuntimeError(
                "no split transcript in context; run split_transcript first"
            )
        golden = NormalizedSubtitle.model_validate_json(
            self.context.split_json_path.read_text(encoding="utf-8")
        )
        for subtitle in self.context.user_subtitles:
            self.bus.log(f"mapping {subtitle.path.name} onto golden transcript ...")
            result = map_transcript(
                golden,
                subtitle,
                language=self.language,
                english_style=self.english_style,
                on_log=lambda message: self.bus.log(message, level="WARNING"),
            )
            output_path = self.context.run_dir / f"{subtitle.path.stem}.mapped.json"
            text = result.mapped.model_dump_json(indent=2, ensure_ascii=False) + "\n"
            output_path.write_text(text, encoding="utf-8")
            report_path = (
                self.context.run_dir / f"{subtitle.path.stem}.mapping_report.txt"
            )
            report_path.write_text(result.report + "\n", encoding="utf-8")
            self.context.mapped_paths.append(output_path)
            self.context.mapping_report_paths.append(report_path)
            self.bus.log(
                f"{subtitle.path.name} mapped: {len(result.mapped.dialogue)} cues; "
                f"report: {report_path.name}"
            )
