from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable


class EventType(Enum):
    LOG = "log"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    PROGRESS = "progress"


@dataclass(frozen=True)
class Event:
    type: EventType
    step: str | None = None
    message: str | None = None
    level: str = "INFO"
    progress: float | None = None
    duration: float | None = None


Listener = Callable[[Event], None]


class EventBus:
    def __init__(self) -> None:
        self._listeners: list[Listener] = []

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def emit(self, event: Event) -> None:
        for listener in self._listeners:
            listener(event)

    def log(self, message: str, level: str = "INFO") -> None:
        self.emit(Event(type=EventType.LOG, message=message, level=level))

    def step_started(self, step: str) -> None:
        self.emit(Event(type=EventType.STEP_STARTED, step=step))

    def step_completed(self, step: str, duration: float | None = None) -> None:
        self.emit(Event(type=EventType.STEP_COMPLETED, step=step, duration=duration))

    def step_failed(
        self, step: str, message: str, duration: float | None = None
    ) -> None:
        self.emit(
            Event(
                type=EventType.STEP_FAILED,
                step=step,
                message=message,
                duration=duration,
            )
        )

    def set_progress(self, step: str, progress: float | None) -> None:
        self.emit(Event(type=EventType.PROGRESS, step=step, progress=progress))
