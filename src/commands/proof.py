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
        force=force,
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
