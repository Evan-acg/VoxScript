from __future__ import annotations

from src.core.events import EventBus
from src.entity.context import PipelineContext
from src.entity.proof import ProofArgs
from src.entity.translate import LLMConfig
from src.llm import create_provider
from src.service.translate import translate_subtitle_file


class TranslateHandler:
    def __init__(
        self,
        args: ProofArgs,
        context: PipelineContext,
        bus: EventBus,
        config: LLMConfig,
        source_style: str = "Eng",
    ) -> None:
        self.args = args
        self.context = context
        self.bus = bus
        self.config = config
        self.source_style = source_style

    def translate(self) -> None:
        if not self.context.mapped_paths:
            raise RuntimeError(
                "no mapped subtitles in context; run map_timeline first"
            )
        api_key = self.args.api_key or ""
        if not api_key:
            raise RuntimeError("no LLM api key resolved for translation")
        try:
            provider = create_provider(self.config, api_key)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

        for path in self.context.mapped_paths:
            self.bus.log(f"translating {path.name} ...")
            output_path = (
                self.context.run_dir / f"{path.stem}.translated.json"
            )
            report_path = (
                self.context.run_dir / f"{path.stem}.translation_report.txt"
            )
            result = translate_subtitle_file(
                path,
                output_path,
                report_path,
                config=self.config,
                provider=provider,
                source_style=self.source_style,
                on_log=lambda message: self.bus.log(message, level="WARNING"),
                on_progress=lambda pct: self.bus.set_progress(
                    "translate_subtitles", pct
                ),
            )
            self.context.translated_paths.append(output_path)
            self.context.translation_report_paths.append(report_path)
            self.bus.log(
                f"{path.name} translated: {len(result.translated.dialogue)} cues; "
                f"report: {report_path.name}"
            )
