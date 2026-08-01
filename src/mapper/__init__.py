from __future__ import annotations

from src.mapper.base import (
    CorrectionParams,
    DeviationCorrector,
    DeviationWindow,
    find_breakpoints,
    mean,
    median,
    pearson,
    std,
)
from src.mapper.cumulative import CumulativeCorrector
from src.mapper.discrete import DiscreteCorrector
from src.mapper.linear import LinearCorrector
from src.mapper.overall import OverallCorrector
from src.mapper.segmented import SegmentedCorrector

_CORRECTORS: tuple[DeviationCorrector, ...] = (
    CumulativeCorrector(),
    SegmentedCorrector(),
    LinearCorrector(),
    OverallCorrector(),
    DiscreteCorrector(),
)

_FALLBACK = OverallCorrector()


def detect_pattern(
    window: DeviationWindow,
    count_mismatch: bool,
    force_overall: bool = False,
) -> tuple[DeviationCorrector, CorrectionParams]:
    if force_overall:
        return _FALLBACK, CorrectionParams(
            pattern="overall", delta=median(window.series)
        )
    for corrector in _CORRECTORS:
        params = corrector.detect(window, count_mismatch)
        if params is not None:
            return corrector, params
    return _FALLBACK, CorrectionParams(pattern="overall", delta=median(window.series))


__all__ = [
    "CorrectionParams",
    "DeviationCorrector",
    "DeviationWindow",
    "detect_pattern",
    "find_breakpoints",
    "mean",
    "median",
    "pearson",
    "std",
]
