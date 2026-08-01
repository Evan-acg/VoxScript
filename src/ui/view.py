from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from rich.console import RenderableType

from src.core.events import Event


class StepStatus(str, Enum):
    PENDING = "pending"
    CURRENT = "current"
    DONE = "done"
    FAILED = "failed"


def format_duration(seconds: float) -> str:
    if seconds >= 60:
        minutes, remainder = divmod(seconds, 60)
        return f"{int(minutes)}m {remainder:.1f}s"
    return f"{seconds:.1f}s"


class PanelView(ABC):
    @abstractmethod
    def handle(self, event: Event) -> None:
        raise NotImplementedError

    @abstractmethod
    def render(self, height: int | None = None) -> RenderableType:
        raise NotImplementedError
