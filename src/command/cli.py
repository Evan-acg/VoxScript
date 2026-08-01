from __future__ import annotations

from pathlib import Path

import click
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from src.command.pipeline import build_pipeline
from src.core.config import AppConfig
from src.core.events import EventBus
from src.entity.proof import ProofArgs
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
@click.option(
    "-l",
    "--language",
    "language",
    type=str,
    default=None,
    help="Transcription language (e.g. zh); defaults to auto-detect.",
)
def cli(
    input_path: Path,
    subtitle_paths: tuple[str, ...],
    output_path: Path | None,
    force_check: bool,
    language: str | None,
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
        pipeline = build_pipeline(
            bus,
            args,
            config,
            force_check,
            language,
            track_selector=dashboard.prompt_track,
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
    normalized = ", ".join(str(p) for p in context.normalized_paths) or "n/a"
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
            "\n",
            ("Transcript: ", "bold"),
            (str(context.transcript_path), "cyan"),
            "\n",
            ("Transcript (normalized): ", "bold"),
            (str(context.transcript_normalized_path), "cyan"),
            "\n",
            ("Transcript (split): ", "bold"),
            (str(context.split_json_path), "cyan"),
            "\n",
            ("Mapped: ", "bold"),
            (", ".join(str(p) for p in context.mapped_paths) or "n/a", "cyan"),
            "\n",
            ("Reports: ", "bold"),
            (
                ", ".join(str(p) for p in context.mapping_report_paths) or "n/a",
                "cyan",
            ),
            "\n",
            ("Normalized: ", "bold"),
            (normalized, "cyan"),
        ),
        title="Done",
        border_style="green",
    )
    console.print(summary)


if __name__ == "__main__":
    cli()
