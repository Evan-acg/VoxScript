from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import click
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from src.command.pipeline import build_pipeline
from src.core.config import AppConfig
from src.core.events import EventBus
from src.entity.context import PipelineContext
from src.entity.proof import ProofArgs
from src.entity.translate import LLMConfig
from src.ui.dashboard import Dashboard
from src.ui.view import format_duration

console = Console()


def _fmt_paths(paths: list[Path]) -> str:
    return ", ".join(str(path) for path in paths) or "n/a"


_STEP_ARTIFACTS: dict[str, list[tuple[str, Callable[[PipelineContext], str]]]] = {
    "extract_audio": [
        ("Audio: ", lambda ctx: str(ctx.audio_path) if ctx.audio_path else "n/a"),
        (
            "Audio track: ",
            lambda ctx: str(ctx.audio_track) if ctx.audio_track is not None else "n/a",
        ),
    ],
    "transcribe": [
        (
            "Transcript: ",
            lambda ctx: str(ctx.transcript_path) if ctx.transcript_path else "n/a",
        ),
    ],
    "normalize_subtitles": [
        (
            "Transcript (normalized): ",
            lambda ctx: (
                str(ctx.transcript_normalized_path)
                if ctx.transcript_normalized_path
                else "n/a"
            ),
        ),
        ("Normalized: ", lambda ctx: _fmt_paths(ctx.normalized_paths)),
    ],
    "split_transcript": [
        (
            "Transcript (split): ",
            lambda ctx: str(ctx.split_json_path) if ctx.split_json_path else "n/a",
        ),
    ],
    "map_timeline": [
        ("Mapped: ", lambda ctx: _fmt_paths(ctx.mapped_paths)),
        ("Reports: ", lambda ctx: _fmt_paths(ctx.mapping_report_paths)),
    ],
    "translate_subtitles": [
        ("Translated: ", lambda ctx: _fmt_paths(ctx.translated_paths)),
        (
            "Translation reports: ",
            lambda ctx: _fmt_paths(ctx.translation_report_paths),
        ),
    ],
}


@click.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Input video or audio file.",
)
@click.option(
    "-s",
    "--subtitle",
    "subtitle_paths",
    multiple=True,
    default=None,
    help="Subtitle file path, repeatable; optional - when omitted, "
    "the output subtitles come entirely from the whisper transcript.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=str,
    default=None,
    help="Output ASS file (must end with .ass), or a directory with a "
    "trailing slash (created if missing) named after the input; "
    "defaults to <input>.ass.",
)
@click.option(
    "--force-check",
    is_flag=True,
    default=False,
    help="Force re-running the preflight checks.",
)
@click.option(
    "-l",
    "--language",
    "language",
    type=str,
    default=None,
    help="Transcription language (e.g. zh); defaults to auto-detect.",
)
@click.option(
    "-t",
    "--translate",
    "translate",
    is_flag=True,
    default=False,
    help="Translate the mapped subtitles with an LLM.",
)
@click.option(
    "--llm-provider",
    type=str,
    default=None,
    help="LLM provider strategy (default: openai).",
)
@click.option(
    "--source-lang",
    type=str,
    default=None,
    help="Source language of the subtitles (e.g. en).",
)
@click.option(
    "--target-lang",
    type=str,
    default=None,
    help="Target language for translation (e.g. zh-CN).",
)
@click.option(
    "--llm-model",
    type=str,
    default=None,
    help="LLM model name (e.g. deepseek-v3).",
)
@click.option(
    "--api-endpoint",
    type=str,
    default=None,
    help="LLM API endpoint URL.",
)
@click.option(
    "--api-key",
    type=str,
    default=None,
    help="LLM API key; falls back to the api_key_env environment variable.",
)
@click.option(
    "--api-key-env",
    type=str,
    default=None,
    help="Environment variable holding the LLM API key (default: LLM_API_KEY).",
)
@click.option(
    "--term-base",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="JSON term base file: {\"source\": \"translation\"}.",
)
@click.option(
    "--max-lines-per-request",
    type=int,
    default=None,
    help="Max subtitle lines per LLM request (default: 30).",
)
@click.option(
    "--temperature",
    type=float,
    default=None,
    help="LLM generation temperature (default: 0.3).",
)
@click.option(
    "--summary",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Text file with the episode summary (macro context).",
)
@click.option(
    "--characters",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Text file describing characters and their speaking styles.",
)
@click.option(
    "--relationships",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Text file describing relationships between characters.",
)
@click.option(
    "--llm-concurrency",
    type=int,
    default=None,
    help="Number of parallel translation requests (default: 4).",
)
@click.option(
    "--sequential",
    is_flag=True,
    default=False,
    help="Translate batches strictly in order (previous translations "
    "used as context) instead of in parallel.",
)
def cli(
    input_path: Path,
    subtitle_paths: tuple[str, ...],
    output_path: Path | None,
    force_check: bool,
    language: str | None,
    translate: bool,
    llm_provider: str | None,
    source_lang: str | None,
    target_lang: str | None,
    llm_model: str | None,
    api_endpoint: str | None,
    api_key: str | None,
    api_key_env: str | None,
    term_base: Path | None,
    max_lines_per_request: int | None,
    temperature: float | None,
    summary: Path | None,
    characters: Path | None,
    relationships: Path | None,
    llm_concurrency: int | None,
    sequential: bool,
) -> None:
    """VoxScript CLI."""
    config = AppConfig.load()
    llm_config = _build_llm_config(
        config,
        translate,
        llm_provider,
        source_lang,
        target_lang,
        llm_model,
        api_endpoint,
        api_key,
        api_key_env,
        term_base,
        max_lines_per_request,
        temperature,
        summary,
        characters,
        relationships,
        llm_concurrency,
        sequential,
    )
    try:
        output_is_dir = bool(
            output_path and output_path.rstrip().endswith(("/", "\\"))
        )
        args = ProofArgs(
            input_path=input_path,
            subtitle_paths=list(subtitle_paths),
            output_path=Path(output_path) if output_path else None,
            output_is_dir=output_is_dir,
            model_dir=config.model_dir,
            api_key=llm_config.api_key if llm_config is not None else None,
            api_key_env=(
                llm_config.api_key_env if llm_config is not None else None
            ),
        )
    except ValidationError as exc:
        raise click.ClickException(exc.errors()[0]["msg"]) from exc

    bus = EventBus()
    dashboard = Dashboard(bus, console=console)
    dashboard.start()
    try:
        pipeline = build_pipeline(
            bus,
            args,
            config,
            force_check,
            language,
            track_selector=dashboard.prompt_track,
            llm_config=llm_config,
        )
        dashboard.set_steps(pipeline.names)
        pipeline.run()
    except click.ClickException:
        dashboard.stop()
        dashboard.print_snapshot()
        raise
    finally:
        dashboard.stop()

    dashboard.print_snapshot()

    context = pipeline.context
    artifacts = {
        **_STEP_ARTIFACTS,
        "export_ass": [("Output: ", lambda ctx: str(args.output_path))],
    }
    parts: list[tuple[str, str]] = []
    for name in pipeline.names:
        for label, getter in artifacts.get(name, []):
            parts.append((label, "bold"))
            parts.append((getter(context), "cyan"))
            parts.append(("\n", ""))
    total = format_duration(sum(pipeline.durations.values()))
    parts.append(("Total time: ", "bold"))
    parts.append((total, "cyan"))
    done_panel = Panel(
        Text.assemble(*parts),
        title="Done",
        border_style="green",
    )
    console.print(done_panel)


def _build_llm_config(
    config: AppConfig,
    translate: bool,
    llm_provider: str | None,
    source_lang: str | None,
    target_lang: str | None,
    llm_model: str | None,
    api_endpoint: str | None,
    api_key: str | None,
    api_key_env: str | None,
    term_base_path: Path | None,
    max_lines_per_request: int | None,
    temperature: float | None,
    summary_path: Path | None,
    characters_path: Path | None,
    relationships_path: Path | None,
    llm_concurrency: int | None,
    sequential: bool,
) -> LLMConfig | None:
    if not translate:
        return None
    base = config.llm or LLMConfig()
    term_base = dict(base.term_base)
    if term_base_path is not None:
        term_base.update(_load_term_base(term_base_path))
    merged = LLMConfig(
        provider=llm_provider or base.provider,
        model=llm_model or base.model,
        api_endpoint=api_endpoint or base.api_endpoint,
        api_key=api_key or "",
        api_key_env=api_key_env or base.api_key_env,
        source_lang=source_lang or base.source_lang,
        target_lang=target_lang or base.target_lang,
        term_base=term_base,
        max_lines_per_request=(
            max_lines_per_request
            if max_lines_per_request is not None
            else base.max_lines_per_request
        ),
        temperature=(
            temperature if temperature is not None else base.temperature
        ),
        summary=(
            summary_path.read_text(encoding="utf-8")
            if summary_path is not None
            else base.summary
        ),
        characters=(
            characters_path.read_text(encoding="utf-8")
            if characters_path is not None
            else base.characters
        ),
        relationships=(
            relationships_path.read_text(encoding="utf-8")
            if relationships_path is not None
            else base.relationships
        ),
        target_style=base.target_style,
        alias_groups=base.alias_groups,
        concurrency=(
            llm_concurrency
            if llm_concurrency is not None
            else base.concurrency
        ),
        sequential=sequential or base.sequential,
    )
    if not merged.model or not merged.api_endpoint:
        raise click.ClickException(
            "LLM translation requires model and api_endpoint: set them under "
            "[llm] in configs/config.yaml or pass --llm-model and --api-endpoint"
        )
    return merged


def _load_term_base(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise click.ClickException(
            f"cannot read term base file: {path} - {exc}"
        ) from exc
    if not isinstance(data, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in data.items()
    ):
        raise click.ClickException(
            f"term base must be a JSON object of string pairs: {path}"
        )
    return data


if __name__ == "__main__":
    cli()
