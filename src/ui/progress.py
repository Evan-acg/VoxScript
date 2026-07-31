from __future__ import annotations

from abc import ABC, abstractmethod

from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text

from src.core.events import Event, EventType
from src.ui.view import PanelView


class ProgressStrategy(ABC):
    def __init__(self, console: Console | None = None) -> None:
        self._console = console

    @abstractmethod
    def update(self, step: str, pct: float | None) -> None:
        raise NotImplementedError

    @abstractmethod
    def render(self) -> RenderableType:
        raise NotImplementedError


class DeterminateProgress(ProgressStrategy):
    def __init__(self, console: Console | None = None) -> None:
        super().__init__(console)
        self._progress = Progress(
            BarColumn(),
            TextColumn("[bold]{task.description}"),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        )
        self._task_id: int | None = None

    def update(self, step: str, pct: float | None) -> None:
        if self._task_id is None:
            self._task_id = self._progress.add_task(
                step, total=100, completed=pct or 0
            )
        else:
            self._progress.update(self._task_id, completed=pct or 0)

    def render(self) -> RenderableType:
        return self._progress


class IndeterminateProgress(ProgressStrategy):
    def __init__(self, console: Console | None = None) -> None:
        super().__init__(console)
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        )
        self._task_id: int | None = None

    def update(self, step: str, pct: float | None) -> None:
        if self._task_id is None:
            self._task_id = self._progress.add_task(step, total=None)

    def render(self) -> RenderableType:
        return self._progress


class ProgressView(PanelView):
    def __init__(self, console: Console | None = None) -> None:
        self._console = console
        self._progress: dict[str, float | None] = {}
        self._current: str | None = None
        self._strategy: ProgressStrategy | None = None
        self._strategy_step: str | None = None
        self._strategy_mode: str | None = None

    def handle(self, event: Event) -> None:
        if event.type is EventType.STEP_STARTED:
            self._current = event.step
        elif event.type is EventType.STEP_COMPLETED:
            self._progress[event.step] = 100.0
            self._current = None
        elif event.type is EventType.STEP_FAILED:
            self._current = None
        elif event.type is EventType.PROGRESS:
            self._progress[event.step] = event.progress

    def render(self, height: int | None = None) -> RenderableType:
        if self._current is None:
            return Panel(
                Text("Idle", style="dim"),
                title="Progress",
                border_style="green",
            )

        pct = self._progress.get(self._current)
        mode = "bar" if pct is not None else "spinner"
        if (
            self._strategy is None
            or self._strategy_step != self._current
            or self._strategy_mode != mode
        ):
            strategy_class = (
                DeterminateProgress if mode == "bar" else IndeterminateProgress
            )
            self._strategy = strategy_class(self._console)
            self._strategy_step = self._current
            self._strategy_mode = mode
        self._strategy.update(self._current, pct)
        return Panel(
            self._strategy.render(),
            title="Progress",
            border_style="green",
        )
