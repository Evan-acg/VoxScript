from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Column

from ..config import get


_STAGE_LABELS: dict[str, str] = {
    "proofread": "校对",
    "whisperx_load": "加载模型",
    "whisperx": "转写中",
    "whisperx_align": "对齐中",
    "whisperx_align_load": "加载对齐模型",
    "ffmpeg": "提取音频",
    "ffprobe": "视频信息",
    "cache": "使用缓存",
    "llm": "LLM校对",
    "voxscript": "输出字幕",
    "audio_check": "检查音频",
    "align_load": "加载对齐模型",
    "align": "强制对齐",
    "alignment_cache": "对齐缓存",
    "gap_asr": "缺失片段转写",
    "transcription_cache": "转录缓存",
}


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    completed: float
    total: float
    message: str = ""


ProgressCallback = Callable[[ProgressEvent], None]


def null_callback(event: ProgressEvent) -> None:
    pass


class RichProgressReporter:
    def __init__(self) -> None:
        cs = get("progress", "color_system", fallback="truecolor")
        self._console = Console(color_system=cs)

        self._progress = Progress(
            TextColumn("[progress.description]{task.description}",
                       table_column=Column(ratio=1, no_wrap=True)),
            BarColumn(table_column=Column(ratio=4)),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%",
                       table_column=Column(ratio=1)),
            TimeElapsedColumn(table_column=Column(ratio=1)),
            expand=True,
            console=self._console,
        )
        self._tasks: dict[str, TaskID] = {}

        self._log_lines: list[str] = []

        self._layout = Layout()
        self._layout.split_column(
            Layout(
                Panel("", title="\u65e5\u5fd7", border_style="dim"),
                name="log",
                ratio=3,
            ),
            Layout(
                Panel(self._progress, title="\u8fdb\u5ea6", border_style="dim"),
                name="progress",
                ratio=1,
                minimum_size=3,
            ),
        )

        self._live: Live | None = None
        self._loguru_handler_id: int | None = None

    def _append_log(self, message: str) -> None:
        self._log_lines.append(message)
        if len(self._log_lines) > 200:
            self._log_lines = self._log_lines[-200:]
        visible = "\n".join(self._log_lines[-15:])
        self._layout["log"].update(
            Panel(visible, title="日志", border_style="dim")
        )

    def __enter__(self) -> RichProgressReporter:
        from loguru import logger

        logger.remove()

        self._loguru_handler_id = logger.add(
            self._append_log,
            level="INFO",
            format="{time:HH:mm:ss} | {level:<7} | {message}",
        )

        for name in [
            "whisperx", "pyannote", "lightning", "lightning.pytorch",
            "huggingface_hub", "transformers", "httpcore", "httpx",
            "openai", "filelock", "torch", "whisper",
            "whisperx.asr", "whisperx.vads",
        ]:
            logging.getLogger(name).handlers.clear()
            logging.getLogger(name).propagate = True

        self._live = Live(
            self._layout,
            console=self._console,
            refresh_per_second=4,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        from loguru import logger

        if self._live is not None:
            self._live.__exit__(*args)
            self._live = None

        if self._loguru_handler_id is not None:
            logger.remove(self._loguru_handler_id)
            self._loguru_handler_id = None

        logger.add(
            sys.stderr,
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
        )

    def as_callback(self) -> ProgressCallback:
        def callback(event: ProgressEvent) -> None:
            if event.stage not in self._tasks:
                self._tasks[event.stage] = self._progress.add_task(
                    event.stage, total=None
                )
            task_id = self._tasks[event.stage]
            is_determinate = event.total > 0
            total: float | None = event.total if is_determinate else None
            desc = _STAGE_LABELS.get(event.stage) or event.message or event.stage
            self._progress.update(
                task_id,
                total=total,
                completed=event.completed if total is not None else 0,
                description=desc,
            )
            if is_determinate and event.completed >= event.total:
                self._progress.remove_task(task_id)
                del self._tasks[event.stage]

        return callback
