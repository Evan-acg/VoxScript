from __future__ import annotations

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from src.core.events import Event, EventType
from src.ui.view import PanelView, StepStatus, format_duration

_ICONS = {
    StepStatus.DONE: ("\u2713", "green"),
    StepStatus.CURRENT: ("\u25c9", "yellow"),
    StepStatus.FAILED: ("\u2717", "red"),
    StepStatus.PENDING: ("\u2022", "dim"),
}


class StepsView(PanelView):
    def __init__(
        self,
        steps: list[str],
        console: Console | None = None,
    ) -> None:
        self._steps = list(steps)
        self._console = console
        self._status: dict[str, StepStatus] = {
            step: StepStatus.PENDING for step in self._steps
        }
        self._durations: dict[str, float] = {}

    def handle(self, event: Event) -> None:
        if event.type is EventType.STEP_STARTED:
            self._status[event.step] = StepStatus.CURRENT
        elif event.type is EventType.STEP_COMPLETED:
            self._status[event.step] = StepStatus.DONE
            if event.duration is not None:
                self._durations[event.step] = event.duration
        elif event.type is EventType.STEP_FAILED:
            self._status[event.step] = StepStatus.FAILED
            if event.duration is not None:
                self._durations[event.step] = event.duration

    def render(self, height: int | None = None) -> RenderableType:
        lines: list[Text] = []
        for step in self._steps:
            status = self._status.get(step, StepStatus.PENDING)
            icon, style = _ICONS[status]
            line = Text()
            line.append(f"{icon} ", style=style)
            line.append(step.replace("_", " ").title(), style=style)
            if step in self._durations:
                line.append(
                    f"  {format_duration(self._durations[step])}", style="dim"
                )
            lines.append(line)
        return Panel(
            Group(*lines),
            title="Steps",
            border_style="cyan",
            expand=True,
        )
