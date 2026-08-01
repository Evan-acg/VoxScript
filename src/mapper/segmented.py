from __future__ import annotations

import bisect

from src.mapper.base import (
    CorrectionParams,
    DeviationCorrector,
    DeviationWindow,
    find_breakpoints,
    median,
)


class SegmentedCorrector(DeviationCorrector):
    match_kind = "sequential"

    def detect(
        self, window: DeviationWindow, count_mismatch: bool
    ) -> CorrectionParams | None:
        breakpoints = find_breakpoints(window.series)
        if not breakpoints:
            return None
        deltas: list[float] = []
        start = 0
        for boundary in breakpoints:
            deltas.append(median(window.series[start:boundary]))
            start = boundary
        deltas.append(median(window.series[start:]))
        return CorrectionParams(
            pattern="segmented",
            breakpoints=tuple(breakpoints),
            segment_deltas=tuple(deltas),
        )

    def correct_times(
        self, entries: list[tuple[float, float]], params: CorrectionParams
    ) -> list[tuple[float, float]]:
        result: list[tuple[float, float]] = []
        for index, (start, end) in enumerate(entries):
            segment = bisect.bisect_right(params.breakpoints, index)
            delta = params.segment_deltas[
                min(segment, len(params.segment_deltas) - 1)
            ]
            result.append((start + delta, end + delta))
        return result
