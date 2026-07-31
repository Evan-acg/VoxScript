from __future__ import annotations

from pathlib import Path

import click

_ALLOWED_FORMATS = ("srt", "ass", "ssa")


def validate_subtitle_path(value: str) -> None:
    suffix = value.rsplit(".", 1)[-1] if "." in value else ""
    if suffix not in _ALLOWED_FORMATS:
        raise click.ClickException(
            f"unsupported subtitle format: {suffix or '<none>'}; "
            f"expected one of: {', '.join(_ALLOWED_FORMATS)}"
        )


@click.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Input video or audio file.",
)
@click.option(
    "-s",
    "--subtitle",
    "subtitle_paths",
    multiple=True,
    required=True,
    help="Subtitle file path, repeatable; braces like {srt, ass, ssa} list allowed formats.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output ASS file; defaults to <input>.ass.",
)
def cli(input_path: Path, subtitle_paths: tuple[str, ...], output_path: Path | None) -> None:
    """VoxScript placeholder CLI."""
    for value in subtitle_paths:
        validate_subtitle_path(value)

    output = output_path or input_path.with_suffix(".ass")

    click.echo(f"Input video: {input_path}")
    for value in subtitle_paths:
        click.echo(f"Subtitle: {value}")
    click.echo(f"Output subtitle: {output}")


if __name__ == "__main__":
    cli()
