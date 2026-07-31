from __future__ import annotations

from rich.console import Console
from rich.layout import Layout

from src.ui.logs import LogView
from src.ui.progress import ProgressView
from src.ui.steps import StepsView

_BAR_HEIGHT = 5
_LEFT_RATIO = 1
_RIGHT_RATIO = 2


class LayoutComposer:
    def __init__(self, console: Console) -> None:
        self._console = console

    def log_region_height(self, console_height: int) -> int:
        main_height = max(1, console_height - _BAR_HEIGHT)
        right_height = int(
            main_height * _RIGHT_RATIO / (_LEFT_RATIO + _RIGHT_RATIO)
        )
        return max(1, right_height)

    def build(
        self,
        steps: StepsView,
        logs: LogView,
        progress: ProgressView,
        console_height: int,
    ) -> Layout:
        layout = Layout()
        layout.split(
            Layout(name="main", ratio=1),
            Layout(name="bar", size=_BAR_HEIGHT),
        )
        layout["main"].split_row(
            Layout(name="left", ratio=_LEFT_RATIO, minimum_size=30),
            Layout(name="right", ratio=_RIGHT_RATIO, minimum_size=50),
        )
        layout["left"].update(steps.render())
        layout["right"].update(
            logs.render(height=self.log_region_height(console_height))
        )
        layout["bar"].update(progress.render())
        return layout
