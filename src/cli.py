from __future__ import annotations

import logging

import click

from audio import FFmpegAudioExtractor, is_audio_only
from config import get, get_section
from llm import LLMClient
from pipeline import Options, VoxScriptPipeline
from progress import RichProgressReporter
from proofreader import Proofreader
from subtitle import SrtFormatter, WhisperXTranscriber

log_cfg = get_section("logging")
logging.basicConfig(
    level=getattr(logging, log_cfg.get("level", "INFO"), logging.INFO),
    format=log_cfg.get("format", "%(message)s"),
)


@click.group()
def cli() -> None:
    """Extract subtitle from video using WhisperX."""


@cli.command()
@click.option(
    "-i",
    "--input",
    "input_file",
    type=click.Path(exists=True),
    required=True,
    help="Input video or audio file",
)
@click.option(
    "-m",
    "--model",
    "model_name",
    default=get("defaults", "model_name", fallback="base"),
    show_default=True,
    type=click.Choice(["tiny", "base", "small", "medium", "large"]),
    help="Whisper model size",
)
@click.option(
    "-l",
    "--language",
    default=get("defaults", "language") or None,
    help="Language code (e.g. zh, en). Auto-detect if not set.",
)
@click.option(
    "-t",
    "--track",
    "track_index",
    default=None,
    type=click.IntRange(0, None),
    help="Audio stream index (0-based). Prompt interactively if not set.",
)
@click.option(
    "-d",
    "--device",
    default=get("defaults", "device", fallback="cuda"),
    show_default=True,
    type=click.Choice(["cpu", "cuda"]),
    help="Device to run inference on",
)
@click.option(
    "-o",
    "--output-dir",
    default=get("defaults", "output_dir", fallback="."),
    show_default=True,
    type=click.Path(file_okay=False),
    help="Output directory for subtitle file",
)
@click.option(
    "--keep-audio",
    is_flag=True,
    default=False,
    help="Keep the extracted audio WAV file",
)
@click.option(
    "--list-tracks",
    is_flag=True,
    default=False,
    help="List available audio streams and exit",
)
def trans(
    input_file: str,
    model_name: str,
    language: str | None,
    track_index: int | None,
    device: str,
    output_dir: str,
    keep_audio: bool,
    list_tracks: bool,
) -> None:
    """Transcribe video/audio to subtitle (SRT)."""
    extractor = FFmpegAudioExtractor()

    if is_audio_only(input_file) and list_tracks:
        click.echo("Input is an audio file, no audio stream selection needed.")
        return

    if list_tracks:
        streams = extractor.list_audio_streams(input_file)
        if not streams:
            click.echo("No audio streams found.")
            return
        click.echo("Available audio tracks:")
        for s in streams:
            ch = "stereo" if s.channels == 2 else "mono"
            click.echo(
                f"  #{s.index}  {s.language or '?':<6}  "
                f"{s.codec:<8}  {s.sample_rate // 1000:>3}kHz  {ch}"
            )
        return

    opts = Options(
        model_name=model_name,
        language=language,
        device=device,
        output_dir=output_dir,
        keep_audio=keep_audio,
        track_index=track_index,
    )

    pipeline = VoxScriptPipeline(
        audio_extractor=extractor,
        transcriber=WhisperXTranscriber(),
        formatter=SrtFormatter(),
    )

    with RichProgressReporter() as reporter:
        pipeline.run(input_file, opts, on_progress=reporter.as_callback())


@cli.command()
@click.option(
    "-i",
    "--input",
    "input_file",
    type=click.Path(exists=True),
    required=True,
    help="Input video or audio file",
)
@click.option(
    "-s",
    "--subtitle",
    "subtitle_file",
    type=click.Path(exists=True),
    required=True,
    help="Subtitle file to proofread (SRT/SSA/ASS)",
)
@click.option(
    "-m",
    "--model",
    "model_name",
    default=get("defaults", "model_name", fallback="base"),
    show_default=True,
    type=click.Choice(["tiny", "base", "small", "medium", "large"]),
    help="Whisper model size for reference transcription",
)
@click.option(
    "-l",
    "--language",
    default=get("defaults", "language") or None,
    help="Language code (e.g. zh, en). Auto-detect if not set.",
)
@click.option(
    "-t",
    "--track",
    "track_index",
    default=None,
    type=click.IntRange(0, None),
    help="Audio stream index (0-based). Prompt interactively if not set.",
)
@click.option(
    "-d",
    "--device",
    default=get("defaults", "device", fallback="cuda"),
    show_default=True,
    type=click.Choice(["cpu", "cuda"]),
    help="Device to run inference on",
)
@click.option(
    "-o",
    "--output-dir",
    default=get("defaults", "output_dir", fallback="."),
    show_default=True,
    type=click.Path(file_okay=False),
    help="Output directory for corrected subtitle",
)
@click.option(
    "--llm-model",
    default="",
    help="LLM model name (overrides config)",
)
@click.option(
    "--keep-audio",
    is_flag=True,
    default=False,
    help="Keep the extracted audio WAV file",
)
def proof(
    input_file: str,
    subtitle_file: str,
    model_name: str,
    language: str | None,
    track_index: int | None,
    device: str,
    output_dir: str,
    llm_model: str,
    keep_audio: bool,
) -> None:
    """Proofread subtitle using LLM against audio transcription (outputs ASS)."""
    model = llm_model or get("llm", "model", fallback="gpt-4o")
    base_url = get("llm", "base_url", fallback="")

    extractor = FFmpegAudioExtractor()
    transcriber = WhisperXTranscriber()
    llm_client = LLMClient(model=model, base_url=base_url)

    opts = Options(
        model_name=model_name,
        language=language,
        device=device,
        output_dir=output_dir,
        keep_audio=keep_audio,
        track_index=track_index,
    )

    proofreader = Proofreader(
        audio_extractor=extractor,
        transcriber=transcriber,
        llm_client=llm_client,
    )

    with RichProgressReporter() as reporter:
        proofreader.run(
            input_file,
            subtitle_file,
            opts,
            on_progress=reporter.as_callback(),
        )
