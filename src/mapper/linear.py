from __future__ import annotations

from src.mapper.base import (
    CorrectionParams,
    DeviationCorrector,
    DeviationWindow,
    pearson,
)


class LinearCorrector(DeviationCorrector):
    match_kind = "sequential"

    def detect(
        self, window: DeviationWindow, count_mismatch: bool
    ) -> CorrectionParams | None:
        correlation = pearson(list(range(len(window.series))), window.series)
        if correlation is None or abs(correlation) <= 0.85:
            return None
        span_b = window.b_end[-1] - window.b_start[0]
        if span_b <= 0:
            return None
        ratio = (window.a_end[-1] - window.a_start[0]) / span_b
        offset = window.a_start[0] - window.b_start[0] * ratio
        return CorrectionParams(pattern="linear", ratio=ratio, offset=offset)

    def correct_times(
        self, entries: list[tuple[float, float]], params: CorrectionParams
    ) -> list[tuple[float, float]]:
        if params.ratio is None or params.offset is None:
            return list(entries)
        return [
            (start * params.ratio + params.offset, end * params.ratio + params.offset)
            for start, end in entries
        ]
