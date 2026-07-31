from __future__ import annotations

from collections import deque

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from src.core.events import Event, EventType
from src.ui.view import PanelView

_LOG_STYLES = {
    "INFO": "default",
    "WARNING": "yellow",
    "ERROR": "red",
}


class LogView(PanelView):
    def __init__(self, console: Console | None = None) -> None:
        self._console = console
        self._log: deque[Text] = deque(maxlen=500)

    def handle(self, event: Event) -> None:
        if event.type is EventType.LOG and event.message:
            style = _LOG_STYLES.get(event.level, "default")
            self._log.append(Text(event.message, style=style))
        elif event.type is EventType.STEP_FAILED and event.message:
            self._log.append(Text(event.message, style="red"))

    def render(self, height: int | None = None) -> RenderableType:
        max_lines = max(1, height - 2) if height else None
        lines = list(self._log)
        if max_lines is not None:
            lines = lines[-max_lines:]
        content: RenderableType
        if lines:
            content = Group(*lines)
        else:
            content = Text("(no log output yet)", style="dim")
        return Panel(content, title="Logs", border_style="blue")
