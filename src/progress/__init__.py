from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from ..config import get


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
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self._console,
        )
        self._tasks: dict[str, TaskID] = {}

    def __enter__(self) -> RichProgressReporter:
        self._progress.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self._progress.__exit__(*args)

    def as_callback(self) -> ProgressCallback:
        def callback(event: ProgressEvent) -> None:
            if event.stage not in self._tasks:
                total: float | None = event.total if event.total > 0 else None
                self._tasks[event.stage] = self._progress.add_task(
                    event.stage, total=total
                )
            task_id = self._tasks[event.stage]
            self._progress.update(
                task_id,
                completed=event.completed,
                description=event.message or event.stage,
            )

        return callback
