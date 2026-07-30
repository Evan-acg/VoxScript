from __future__ import annotations

from pathlib import Path

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
    default=get("defaults", "model_name", fallback="medium"),
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

force_option = click.option(
    "-f",
    "--force",
    is_flag=True,
    default=False,
    help="Force re-extract audio and re-generate subtitle",
)

ss_option = click.option(
    "--ss",
    type=str,
    default=None,
    help="Start time (HH:MM:SS)",
)

to_option = click.option(
    "--to",
    type=str,
    default=None,
    help="End time (HH:MM:SS), requires --ss",
)


def parse_hms(s: str) -> float:
    parts = list(map(int, s.split(":")))
    if len(parts) == 3:
        return float(parts[0] * 3600 + parts[1] * 60 + parts[2])
    raise click.BadParameter("time must be HH:MM:SS")


def resolve_output_dir(
    input_file: str,
    output_dir: str,
    subtitle_file: str | None = None,
) -> str:
    default = get("defaults", "output_dir", fallback="./output")
    if output_dir and str(output_dir) != default:
        return str(output_dir)
    video_dir = Path(input_file).parent
    if video_dir.is_dir():
        return str(video_dir)
    if subtitle_file:
        sub_dir = Path(subtitle_file).parent
        if sub_dir.is_dir():
            return str(sub_dir)
    return default
