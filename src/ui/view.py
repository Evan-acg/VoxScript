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


class PanelView(ABC):
    @abstractmethod
    def handle(self, event: Event) -> None:
        raise NotImplementedError

    @abstractmethod
    def render(self, height: int | None = None) -> RenderableType:
        raise NotImplementedError
