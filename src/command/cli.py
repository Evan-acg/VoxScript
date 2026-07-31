from __future__ import annotations

from pathlib import Path

import click
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from src.core.config import AppConfig
from src.core.preflight import run_preflight
from src.entity.context import PipelineContext
from src.entity.proof import ProofArgs
from src.handler.audio import AudioHandler

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
def cli(
    input_path: Path,
    subtitle_paths: tuple[str, ...],
    output_path: Path | None,
) -> None:
    """VoxScript CLI."""
    config = AppConfig.load()
    results = run_preflight(config)
    table = Table(title="Preflight checks")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for result in results:
        status = "[green]OK[/green]" if result.ok else "[red]FAILED[/red]"
        table.add_row(result.name, status, result.detail)
    console.print(table)
    if not all(result.ok for result in results):
        failed = [result.name for result in results if not result.ok]
        raise click.ClickException(f"preflight checks failed: {', '.join(failed)}")

    try:
        args = ProofArgs(
            input_path=input_path,
            subtitle_paths=list(subtitle_paths),
            output_path=output_path,
            model_dir=config.model_dir,
        )
    except ValidationError as exc:
        raise click.ClickException(exc.errors()[0]["msg"]) from exc

    click.echo(f"Input video: {args.input_path}")
    for value in args.subtitle_paths:
        click.echo(f"Subtitle: {value}")
    click.echo(f"Output subtitle: {args.output_path}")
    click.echo(f"Model dir: {args.model_dir}")

    try:
        context = AudioHandler(args, PipelineContext()).extract_audio()
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Extracted audio: {context.audio_path}")


if __name__ == "__main__":
    cli()
