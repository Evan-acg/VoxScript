from __future__ import annotations

from src.mapper.base import (
    CorrectionParams,
    DeviationCorrector,
    DeviationWindow,
)


class CumulativeCorrector(DeviationCorrector):
    match_kind = "overlap"
    min_step = 0.5
    max_step = 2.0
    min_run = 4

    def detect(
        self, window: DeviationWindow, count_mismatch: bool
    ) -> CorrectionParams | None:
        if not count_mismatch:
            return None
        run = 0
        for index in range(len(window.series) - 1):
            step = window.series[index + 1] - window.series[index]
            if self.min_step <= step <= self.max_step:
                run += 1
                if run >= self.min_run:
                    return CorrectionParams(pattern="cumulative")
            else:
                run = 0
        return None

    def correct_times(
        self, entries: list[tuple[float, float]], params: CorrectionParams
    ) -> list[tuple[float, float]]:
        return list(entries)
