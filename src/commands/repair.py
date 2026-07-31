from __future__ import annotations

from pathlib import Path

import click

from ..config import get
from ..progress import RichProgressReporter
from ..repair.asr import WhisperXASR
from ..repair.llm import LLMClient
from ..repair.media import FFmpegAudioExtractor
from ..repair.workflow import RepairOptions, run_repair


@click.command()
@click.option(
    "--video",
    "video_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Input video or audio file.",
)
@click.option(
    "--subtitle",
    "subtitle_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Original ASS subtitle file.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Final output ASS file; defaults to <video>.repaired.ass.",
)
@click.option(
    "--chunk-minutes",
    type=click.FloatRange(min=0.1),
    default=10.0,
    show_default=True,
    help="Length of each body chunk in minutes.",
)
@click.option(
    "--context-seconds",
    type=click.FloatRange(min=0.0),
    default=10.0,
    show_default=True,
    help="Context kept on both sides of each chunk.",
)
@click.option(
    "--model",
    "model_name",
    type=click.Choice(["tiny", "base", "small", "medium", "large"]),
    default=lambda: get("defaults", "model_name", fallback="small"),
    show_default=True,
    help="WhisperX model size.",
)
@click.option(
    "--language",
    default=lambda: get("defaults", "language", fallback="") or None,
    help="Source language for ASR; auto-detect when omitted.",
)
@click.option(
    "--target-language",
    default="auto",
    show_default=True,
    help="Target language of the existing subtitle text.",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"]),
    default=lambda: get("defaults", "device", fallback="cuda"),
    show_default=True,
    help="Device used by WhisperX.",
)
@click.option(
    "--vad-method",
    type=click.Choice(["silero", "pyannote"]),
    default=lambda: get("whisper", "vad_method", fallback="silero"),
    show_default=True,
    help="Voice activity detection backend.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=lambda: get("whisper", "batch_size", fallback="16"),
    show_default=True,
    help="WhisperX inference batch size; increase only when VRAM allows.",
)
@click.option(
    "--asr-chunk-seconds",
    type=click.IntRange(min=1),
    default=lambda: get("whisper", "chunk_size", fallback="5"),
    show_default=True,
    help="Raw ASR VAD chunk size before LLM matching.",
)
@click.option(
    "--align",
    is_flag=True,
    help="Run the slower WhisperX alignment pass for tighter timestamps.",
)
@click.option(
    "--track",
    "track_index",
    type=click.IntRange(min=0),
    default=None,
    help="FFmpeg audio stream index; the first stream is used by default.",
)
@click.option(
    "--llm-model",
    default=lambda: get("llm", "model", fallback="gpt-4o"),
    show_default=True,
    help="OpenAI-compatible LLM model.",
)
@click.option(
    "--work-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory for asr.json, review.json and preview.ass.",
)
@click.option(
    "--keep-artifacts",
    is_flag=True,
    help="Keep intermediate JSON and preview files after success.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-extract chunk audio even when a temporary file exists.",
)
def repair(
    video_path: Path,
    subtitle_path: Path,
    output_path: Path | None,
    chunk_minutes: float,
    context_seconds: float,
    model_name: str,
    language: str | None,
    target_language: str,
    device: str,
    vad_method: str,
    batch_size: int,
    asr_chunk_seconds: int,
    align: bool,
    track_index: int | None,
    llm_model: str,
    work_dir: Path | None,
    keep_artifacts: bool,
    force: bool,
) -> None:
    output = output_path or video_path.with_name(f"{video_path.stem}.repaired.ass")
    base_url = get("llm", "base_url", fallback="")
    try:
        with RichProgressReporter() as reporter:
            result = run_repair(
                video_path,
                subtitle_path,
                output,
                audio=FFmpegAudioExtractor(),
                asr=WhisperXASR(),
                llm=LLMClient(model=llm_model, base_url=base_url),
                options=RepairOptions(
                    chunk_minutes=chunk_minutes,
                    context_seconds=context_seconds,
                    model_name=model_name,
                    language=language or None,
                    target_language=target_language,
                    device=device,
                    vad_method=vad_method,
                    batch_size=batch_size,
                    asr_chunk_seconds=asr_chunk_seconds,
                    align=align,
                    track_index=track_index,
                    force=force,
                    work_dir=work_dir,
                    keep_artifacts=keep_artifacts,
                ),
                on_progress=reporter.as_callback(),
            )
    except Exception as error:
        raise click.ClickException(str(error)) from error

    if result.partial:
        click.echo(
            f"Repaired subtitle saved with conservative fallbacks: {result.output_path}",
            err=True,
        )
    else:
        click.echo(f"Repaired subtitle saved to: {result.output_path}")
    if result.review_path is not None:
        click.echo(f"Automatic report saved to: {result.review_path}")
