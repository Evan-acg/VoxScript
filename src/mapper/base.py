from __future__ import annotations

import math
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class DeviationWindow:
    series: list[float]
    a_start: list[float]
    a_end: list[float]
    b_start: list[float]
    b_end: list[float]


@dataclass(frozen=True)
class CorrectionParams:
    pattern: str
    delta: float = 0.0
    ratio: float | None = None
    offset: float | None = None
    breakpoints: tuple[int, ...] = ()
    segment_deltas: tuple[float, ...] = ()


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values)


def median(values: Sequence[float]) -> float:
    return statistics.median(values)


def std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None
    x_mean, y_mean = mean(x), mean(y)
    x_var = sum((v - x_mean) ** 2 for v in x)
    y_var = sum((v - y_mean) ** 2 for v in y)
    if x_var <= 0 or y_var <= 0:
        return None
    cov = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True))
    return cov / math.sqrt(x_var * y_var)


def find_breakpoints(
    series: Sequence[float],
    jump: float = 1.0,
    max_std: float = 0.3,
    min_side: int = 3,
) -> list[int]:
    n = len(series)
    if n < min_side * 2:
        return []
    candidates = sorted(
        (
            (abs(series[i + 1] - series[i]), i + 1)
            for i in range(n - 1)
            if abs(series[i + 1] - series[i]) > jump
        ),
        reverse=True,
    )
    breaks: list[int] = []
    for _, position in candidates:
        trial = sorted([*breaks, position])
        if all(
            _segment_ok(series, lo, hi, max_std, min_side)
            for lo, hi in _segment_ranges(n, trial)
        ):
            breaks = trial
    return breaks


def _segment_ranges(n: int, breaks: list[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for boundary in breaks:
        ranges.append((start, boundary))
        start = boundary
    ranges.append((start, n))
    return ranges


def _segment_ok(
    series: Sequence[float], lo: int, hi: int, max_std: float, min_side: int
) -> bool:
    if hi - lo < min_side:
        return False
    return std(series[lo:hi]) < max_std


class DeviationCorrector(ABC):
    match_kind: str = "sequential"

    @abstractmethod
    def detect(
        self, window: DeviationWindow, count_mismatch: bool
    ) -> CorrectionParams | None:
        """Return correction params if this pattern applies, else None."""

    @abstractmethod
    def correct_times(
        self, entries: list[tuple[float, float]], params: CorrectionParams
    ) -> list[tuple[float, float]]:
        """Transform (start, end) pairs for all B entries."""
