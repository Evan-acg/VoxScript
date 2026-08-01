from __future__ import annotations

from abc import ABC, abstractmethod

from src.entity.subtitle import SubtitleSegment


class SubtitleParser(ABC):
    @abstractmethod
    def parse(self, content: str) -> list[SubtitleSegment]:
        """Parse subtitle file content into normalized segments."""
