from __future__ import annotations

from pathlib import Path

import click
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from src.core.config import AppConfig
from src.core.events import EventBus
from src.core.preflight import run_preflight_if_needed
from src.entity.context import PipelineContext
from src.entity.proof import ProofArgs
from src.handler.audio import AudioHandler
from src.ui.dashboard import Dashboard

console = Console()


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
    required=True,
    help="Subtitle file path, repeatable.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output ASS file; defaults to <input>.ass.",
)
@click.option(
    "--force-check",
    is_flag=True,
    default=False,
    help="Force re-running the preflight checks.",
)
def cli(
    input_path: Path,
    subtitle_paths: tuple[str, ...],
    output_path: Path | None,
    force_check: bool,
) -> None:
    """VoxScript CLI."""
    config = AppConfig.load()
    try:
        args = ProofArgs(
            input_path=input_path,
            subtitle_paths=list(subtitle_paths),
            output_path=output_path,
            model_dir=config.model_dir,
        )
    except ValidationError as exc:
        raise click.ClickException(exc.errors()[0]["msg"]) from exc

    bus = EventBus()
    dashboard = Dashboard(bus, console=console)
    dashboard.start()
    try:
        context = _run_pipeline(bus, dashboard, args, config, force_check)
    except click.ClickException:
        dashboard.stop()
        dashboard.print_snapshot()
        raise
    finally:
        dashboard.stop()

    dashboard.print_snapshot()

    summary = Panel(
        Text.assemble(
            ("Output: ", "bold"),
            (str(args.output_path), "cyan"),
            "\n",
            ("Audio: ", "bold"),
            (str(context.audio_path), "cyan"),
            "\n",
            ("Audio track: ", "bold"),
            (str(context.audio_track), "cyan"),
        ),
        title="Done",
        border_style="green",
    )
    console.print(summary)


def _run_pipeline(
    bus: EventBus,
    dashboard: Dashboard,
    args: ProofArgs,
    config: AppConfig,
    force_check: bool,
) -> PipelineContext:
    bus.step_started("preflight")
    results = run_preflight_if_needed(config, force=force_check)
    if results is None:
        bus.log("preflight already verified, skipping checks")
    else:
        for result in results:
            status = "OK" if result.ok else "FAILED"
            level = "INFO" if result.ok else "ERROR"
            bus.log(f"{result.name}: {status} - {result.detail}", level=level)
        if not all(result.ok for result in results):
            failed = [result.name for result in results if not result.ok]
            message = f"preflight checks failed: {', '.join(failed)}"
            bus.step_failed("preflight", message)
            raise click.ClickException(message)
    bus.step_completed("preflight")

    context = PipelineContext()
    bus.step_started("extract_audio")
    try:
        context = AudioHandler(
            args,
            context,
            bus,
            track_selector=dashboard.prompt_track,
        ).extract_audio()
    except RuntimeError as exc:
        bus.step_failed("extract_audio", str(exc))
        raise click.ClickException(str(exc)) from exc
    bus.step_completed("extract_audio")
    return context


if __name__ == "__main__":
    cli()
