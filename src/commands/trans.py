from __future__ import annotations

import click

from ..pipeline import Options, VoxScriptPipeline
from ..pipeline.audio import FFmpegAudioExtractor, is_audio_only
from ..pipeline.subtitle import SrtFormatter, WhisperXTranscriber
from ..progress import RichProgressReporter
from .common import (
    device_option,
    input_option,
    keep_audio_option,
    language_option,
    model_option,
    output_dir_option,
    track_option,
)


@click.command()
@input_option
@model_option
@language_option
@track_option
@device_option
@output_dir_option
@keep_audio_option
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
