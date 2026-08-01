from __future__ import annotations

from abc import ABC, abstractmethod

from src.entity.subtitle import SubtitleSegment


class SubtitleParser(ABC):
    @abstractmethod
    def parse(
        self, content: str
    ) -> tuple[list[SubtitleSegment], dict[str, dict[str, str]]]:
        """Parse subtitle file content into normalized segments and a style table."""
