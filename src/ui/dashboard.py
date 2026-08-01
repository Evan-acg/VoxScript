from __future__ import annotations

from typing import Any

from rich.columns import Columns
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live

from src.core.events import Event, EventBus
from src.ui.layout import LayoutComposer
from src.ui.logs import LogView
from src.ui.progress import ProgressView
from src.ui.prompt import TrackPrompt
from src.ui.steps import StepsView


class Dashboard:
    def __init__(
        self,
        bus: EventBus,
        steps: list[str] | None = None,
        console: Console | None = None,
    ) -> None:
        self._console = console or Console()
        self._steps_view = StepsView(steps or [], console=self._console)
        self._logs_view = LogView(console=self._console)
        self._progress_view = ProgressView(console=self._console)
        self._composer = LayoutComposer(self._console)
        self._prompt = TrackPrompt(self._console)
        self._live: Live | None = None
        bus.subscribe(self._on_event)

    def set_steps(self, steps: list[str]) -> None:
        self._steps_view = StepsView(steps, console=self._console)

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

    def pause(self) -> None:
        self.stop()

    def resume(self) -> None:
        self.start()

    def print_snapshot(self) -> None:
        columns = Columns(
            [
                self._steps_view.render(),
                self._logs_view.render(
                    height=self._composer.log_region_height(self._console.size.height)
                ),
            ],
            expand=True,
        )
        self._console.print(Group(columns, self._progress_view.render()))

    def prompt_track(self, streams: list[dict[str, Any]]) -> int:
        return self._prompt.ask(streams, pause=self.pause, resume=self.resume)

    def _on_event(self, event: Event) -> None:
        for view in (self._steps_view, self._logs_view, self._progress_view):
            view.handle(event)
        if self._live is not None:
            self._live.update(self._build())

    def _build(self) -> Layout:
        return self._composer.build(
            self._steps_view,
            self._logs_view,
            self._progress_view,
            self._console.size.height,
        )
