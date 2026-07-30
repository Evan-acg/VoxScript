from __future__ import annotations

import click

from ..config import get

input_option = click.option(
    "-i",
    "--input",
    "input_file",
    type=click.Path(exists=True),
    required=True,
    help="Input video or audio file",
)

model_option = click.option(
    "-m",
    "--model",
    "model_name",
    default=get("defaults", "model_name", fallback="base"),
    show_default=True,
    type=click.Choice(["tiny", "base", "small", "medium", "large"]),
    help="Whisper model size",
)

language_option = click.option(
    "-l",
    "--language",
    default=get("defaults", "language") or None,
    help="Language code (e.g. zh, en). Auto-detect if not set.",
)

track_option = click.option(
    "-t",
    "--track",
    "track_index",
    default=None,
    type=click.IntRange(0, None),
    help="Audio stream index (0-based). Prompt interactively if not set.",
)

device_option = click.option(
    "-d",
    "--device",
    default=get("defaults", "device", fallback="cuda"),
    show_default=True,
    type=click.Choice(["cpu", "cuda"]),
    help="Device to run inference on",
)

output_dir_option = click.option(
    "-o",
    "--output-dir",
    default=get("defaults", "output_dir", fallback="."),
    show_default=True,
    type=click.Path(file_okay=False),
    help="Output directory for subtitle file",
)

keep_audio_option = click.option(
    "--keep-audio",
    is_flag=True,
    default=False,
    help="Keep the extracted audio WAV file",
)
