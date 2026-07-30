from __future__ import annotations

import logging

import click

from .config import get_section

log_cfg = get_section("logging")
logging.basicConfig(
    level=getattr(logging, log_cfg.get("level", "INFO"), logging.INFO),
    format=log_cfg.get("format", "%(message)s"),
)


@click.group()
def cli() -> None:
    """Extract subtitle from video using WhisperX."""


from .commands.trans import trans
from .commands.proof import proof

cli.add_command(trans)
cli.add_command(proof)
