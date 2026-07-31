from __future__ import annotations

import logging
import os
import sys
import warnings

import click
from loguru import logger

from .config import get_section

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

warnings.filterwarnings("ignore", message="torchcodec is not installed")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", message="gradient_checkpointing")
warnings.filterwarnings("ignore", message="TensorFloat-32")

log_cfg = get_section("logging")
logger.remove()
logger.add(
    sys.stderr,
    level=log_cfg.get("level", "INFO"),
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
)


class _InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        logger_opt = logger.opt(depth=6, exception=record.exc_info)
        logger_opt.log(record.levelname, record.getMessage())


logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

for name in [
    "whisperx", "pyannote", "lightning", "lightning.pytorch",
    "huggingface_hub", "transformers", "httpcore", "httpx",
    "openai", "filelock", "torch", "whisper",
]:
    logging.getLogger(name).setLevel(logging.WARNING)


@click.group()
def cli() -> None:
    """Automatically repair ASS subtitles with ASR and an LLM."""


from .commands.repair import repair

cli.add_command(repair)
