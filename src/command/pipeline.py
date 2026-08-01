from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import click

from src.core.config import AppConfig
from src.core.events import EventBus
from src.core.preflight import run_preflight_if_needed
from src.entity.context import PipelineContext
from src.entity.proof import ProofArgs
from src.handler.asr import AsrHandler
from src.handler.audio import AudioHandler, TrackSelector
from src.handler.subtitle import SubtitleHandler


@dataclass(frozen=True)
class Step:
    name: str
    run: Callable[[], PipelineContext | None]


class Pipeline:
    def __init__(self, bus: EventBus, context: PipelineContext) -> None:
        self._bus = bus
        self._context = context
        self._steps: list[Step] = []

    def add(self, name: str, run: Callable[[], PipelineContext | None]) -> "Pipeline":
        self._steps.append(Step(name, run))
        return self

    @property
    def names(self) -> list[str]:
        return [step.name for step in self._steps]

    @property
    def context(self) -> PipelineContext:
        return self._context

    def run(self) -> None:
        for step in self._steps:
            self._bus.step_started(step.name)
            try:
                step.run()
            except RuntimeError as exc:
                message = str(exc)
                self._bus.step_failed(step.name, message)
                raise click.ClickException(message) from exc
            except click.ClickException as exc:
                self._bus.step_failed(step.name, str(exc))
                raise
            self._bus.step_completed(step.name)


def build_pipeline(
    bus: EventBus,
    args: ProofArgs,
    config: AppConfig,
    force_check: bool,
    language: str | None,
    track_selector: TrackSelector | None = None,
) -> Pipeline:
    context = PipelineContext(work_dir=config.work_dir)
    asr = AsrHandler(
        args,
        context,
        bus,
        model_dir=config.model_dir,
        model_name=config.model_name,
        language=language,
    )

    def preflight() -> None:
        results = run_preflight_if_needed(config, force=force_check)
        if results is None:
            bus.log("preflight already verified, skipping checks")
            return
        for result in results:
            status = "OK" if result.ok else "FAILED"
            level = "INFO" if result.ok else "ERROR"
            bus.log(f"{result.name}: {status} - {result.detail}", level=level)
        if not all(result.ok for result in results):
            failed = [result.name for result in results if not result.ok]
            raise click.ClickException(f"preflight checks failed: {', '.join(failed)}")

    def extract_audio() -> None:
        AudioHandler(args, context, bus, track_selector=track_selector).extract_audio()

    def normalize_subtitles() -> None:
        SubtitleHandler(args, context, bus).normalize()

    return (
        Pipeline(bus, context)
        .add("preflight", preflight)
        .add("extract_audio", extract_audio)
        .add("load_model", asr.load_model)
        .add("transcribe", asr.transcribe)
        .add("normalize_subtitles", normalize_subtitles)
    )
