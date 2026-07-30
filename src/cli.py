from __future__ import annotations

import logging
import os
import sys

import click
from loguru import logger

from .config import get_section

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS", "1")

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
    "whisperx",
    "pyannote",
    "lightning",
    "huggingface_hub",
    "transformers",
    "httpcore",
    "httpx",
    "openai",
    "filelock",
]:
    logging.getLogger(name).setLevel(logging.WARNING)


@click.group()
def cli() -> None:
    """Extract subtitle from video using WhisperX."""


from .commands.trans import trans
from .commands.proof import proof

cli.add_command(trans)
cli.add_command(proof)
