from __future__ import annotations

from pathlib import Path

import click
from pydantic import ValidationError

from src.core.config import AppConfig
from src.entity.proof import ProofArgs


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


if __name__ == "__main__":
    cli()
