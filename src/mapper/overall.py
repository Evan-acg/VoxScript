from __future__ import annotations

from src.mapper.base import (
    CorrectionParams,
    DeviationCorrector,
    DeviationWindow,
    median,
    std,
)


class OverallCorrector(DeviationCorrector):
    match_kind = "sequential"

    def detect(
        self, window: DeviationWindow, count_mismatch: bool
    ) -> CorrectionParams | None:
        if std(window.series) >= 0.3:
            return None
        return CorrectionParams(pattern="overall", delta=median(window.series))

    def correct_times(
        self, entries: list[tuple[float, float]], params: CorrectionParams
    ) -> list[tuple[float, float]]:
        return [(start + params.delta, end + params.delta) for start, end in entries]
