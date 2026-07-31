from __future__ import annotations

from collections import deque
from typing import Any

from rich.console import Console, Group, RenderableType
from rich.columns import Columns
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import IntPrompt
from rich.table import Table
from rich.text import Text

from src.core.events import Event, EventBus, EventType

PIPELINE_STEPS = ["preflight", "extract_audio", "transcribe", "generate_ass"]

STATUS_PENDING = "pending"
STATUS_CURRENT = "current"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

_LOG_STYLES = {
    "INFO": "default",
    "WARNING": "yellow",
    "ERROR": "red",
}


class Dashboard:
    def __init__(
        self,
        bus: EventBus,
        steps: list[str] | None = None,
        console: Console | None = None,
    ) -> None:
        self._bus = bus
        self._steps = list(steps or PIPELINE_STEPS)
        self._console = console or Console()
        self._live: Live | None = None
        self._step_status: dict[str, str] = {
            step: STATUS_PENDING for step in self._steps
        }
        self._log: deque[Text] = deque(maxlen=500)
        self._progress: dict[str, float | None] = {}
        self._bar_step: str | None = None
        self._bar_mode: str | None = None
        self._bar_widget: Progress | None = None
        self._bar_task: int | None = None
        bus.subscribe(self._on_event)

    def start(self) -> None:
        self._live = Live(
            self._build(),
            console=self._console,
            screen=True,
            refresh_per_second=8,
        )
        self._live.start()

    def stop(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def print_snapshot(self) -> None:
        left = self._build_log()
        right = Group(self._build_steps(), self._build_progress_bar())
        self._console.print(Columns([left, right], expand=True))

    def prompt_track(self, streams: list[dict[str, Any]]) -> int:
        self.stop()
        self._console.print("Multiple audio tracks found:")
        table = Table(title="Audio tracks")
        table.add_column("#", justify="right")
        table.add_column("Index")
        table.add_column("Codec")
        table.add_column("Channels")
        table.add_column("Language")
        for ordinal, stream in enumerate(streams, start=1):
            table.add_row(
                str(ordinal),
                str(stream.get("index")),
                str(stream.get("codec_name", "-")),
                str(stream.get("channels", "-")),
                str(stream.get("tags", {}).get("language", "-")),
            )
        self._console.print(table)
        count = len(streams)
        while True:
            choice = IntPrompt.ask(
                f"Select audio track [1-{count}]",
                default=1,
                show_default=False,
                console=self._console,
            )
            if 1 <= choice <= count:
                break
            self._console.print(
                f"[red]Invalid selection: {choice}. Expected 1-{count}.[/red]"
            )
        self.start()
        return streams[choice - 1]["index"]

    def _on_event(self, event: Event) -> None:
        if event.type is EventType.LOG and event.message:
            style = _LOG_STYLES.get(event.level, "default")
            self._log.append(Text(event.message, style=style))
        elif event.type is EventType.STEP_STARTED:
            self._step_status[event.step] = STATUS_CURRENT
        elif event.type is EventType.STEP_COMPLETED:
            self._step_status[event.step] = STATUS_DONE
            self._progress[event.step] = 100.0
        elif event.type is EventType.STEP_FAILED:
            self._step_status[event.step] = STATUS_FAILED
            if event.message:
                self._log.append(Text(event.message, style="red"))
        elif event.type is EventType.PROGRESS:
            self._progress[event.step] = event.progress

        if self._live is not None:
            self._live.update(self._build())

    def _build(self) -> Layout:
        layout = Layout()
        layout.split_row(
            Layout(name="log", ratio=2, minimum_size=50),
            Layout(name="progress", ratio=1, minimum_size=30),
        )
        layout["progress"].split_column(
            Layout(name="steps", ratio=1),
            Layout(name="bar", size=5),
        )
        layout["log"].update(self._build_log())
        layout["steps"].update(self._build_steps())
        layout["bar"].update(self._build_progress_bar())
        return layout

    def _build_log(self) -> Panel:
        height = self._console.size.height
        max_lines = max(1, height - 3)
        lines = list(self._log)[-max_lines:]
        content: RenderableType
        if lines:
            content = Group(*lines)
        else:
            content = Text("(no log output yet)", style="dim")
        return Panel(content, title="Logs", border_style="blue")

    def _build_steps(self) -> Panel:
        lines: list[Text] = []
        for step in self._steps:
            status = self._step_status.get(step, STATUS_PENDING)
            if status == STATUS_DONE:
                icon, style = "\u2713", "green"
            elif status == STATUS_CURRENT:
                icon, style = "\u25c9", "yellow"
            elif status == STATUS_FAILED:
                icon, style = "\u2717", "red"
            else:
                icon, style = "\u2022", "dim"
            line = Text()
            line.append(f"{icon} ", style=style)
            line.append(step.replace("_", " ").title(), style=style)
            lines.append(line)
        return Panel(Group(*lines), title="Steps", border_style="cyan")

    def _build_progress_bar(self) -> Panel:
        current = self._current_step()
        if current is None:
            return Panel(Text("Idle", style="dim"), title="Progress", border_style="green")

        pct = self._progress.get(current)
        mode = "bar" if pct is not None else "spinner"
        if (
            self._bar_step != current
            or self._bar_mode != mode
            or self._bar_widget is None
        ):
            if mode == "bar":
                progress = Progress(
                    BarColumn(),
                    TextColumn("[bold]{task.description}"),
                    TextColumn("{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                    console=self._console,
                )
            else:
                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[bold]{task.description}"),
                    TimeElapsedColumn(),
                    console=self._console,
                )
            self._bar_widget = progress
            self._bar_step = current
            self._bar_mode = mode
            self._bar_task = progress.add_task(
                current,
                total=100 if mode == "bar" else None,
                completed=pct if pct is not None else 0,
            )
        elif pct is not None and self._bar_task is not None:
            self._bar_widget.update(self._bar_task, completed=pct)
        return Panel(self._bar_widget, title="Progress", border_style="green")

    def _current_step(self) -> str | None:
        for step in self._steps:
            if self._step_status.get(step) == STATUS_CURRENT:
                return step
        return None
