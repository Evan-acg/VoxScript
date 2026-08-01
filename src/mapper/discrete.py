from __future__ import annotations

from src.mapper.base import (
    CorrectionParams,
    DeviationCorrector,
    DeviationWindow,
    std,
)


class DiscreteCorrector(DeviationCorrector):
    match_kind = "overlap"

    def detect(
        self, window: DeviationWindow, count_mismatch: bool
    ) -> CorrectionParams | None:
        if std(window.series) <= 0.5:
            return None
        return CorrectionParams(pattern="discrete")

    def correct_times(
        self, entries: list[tuple[float, float]], params: CorrectionParams
    ) -> list[tuple[float, float]]:
        return list(entries)
