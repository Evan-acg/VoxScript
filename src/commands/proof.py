from __future__ import annotations

import click

from ..config import get
from ..pipeline import Options
from ..pipeline.audio import FFmpegAudioExtractor
from ..pipeline.llm import LLMClient
from ..pipeline.proofreader import Proofreader
from ..pipeline.subtitle import WhisperXTranscriber
from ..progress import RichProgressReporter
from .common import (
    device_option,
    force_option,
    input_option,
    keep_audio_option,
    language_option,
    model_option,
    output_dir_option,
    parse_hms,
    resolve_output_dir,
    ss_option,
    to_option,
    track_option,
)


@click.command()
@input_option
@click.option(
    "-s",
    "--subtitle",
    "subtitle_file",
    type=click.Path(exists=True),
    required=True,
    help="Subtitle file to proofread (SRT/SSA/ASS)",
)
@model_option
@language_option
@track_option
@device_option
@output_dir_option
@click.option(
    "--llm-model",
    default="",
    help="LLM model name (overrides config)",
)
@keep_audio_option
@force_option
@ss_option
@to_option
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
    force: bool,
    ss: str | None,
    to: str | None,
) -> None:
    """Proofread subtitle using LLM against audio transcription (outputs ASS)."""
    if to is not None and ss is None:
        raise click.UsageError("--to requires --ss")

    model = llm_model or get("llm", "model", fallback="gpt-4o")
    base_url = get("llm", "base_url", fallback="")

    extractor = FFmpegAudioExtractor()
    transcriber = WhisperXTranscriber()
    llm_client = LLMClient(model=model, base_url=base_url)

    output_dir = resolve_output_dir(input_file, output_dir, subtitle_file)

    opts = Options(
        model_name=model_name,
        language=language,
        device=device,
        output_dir=output_dir,
        keep_audio=keep_audio,
        track_index=track_index,
        force=force,
        ss=parse_hms(ss) if ss else None,
        to=parse_hms(to) if to else None,
    )

    proofreader = Proofreader(
        audio_extractor=extractor,
        transcriber=transcriber,
        llm_client=llm_client,
    )

    with RichProgressReporter() as reporter:
        result = proofreader.run(
            input_file,
            subtitle_file,
            opts,
            on_progress=reporter.as_callback(),
        )

    print(f"\nResult saved to: {result}")
