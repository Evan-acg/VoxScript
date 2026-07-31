from __future__ import annotations

import re
from collections.abc import Iterable
from difflib import SequenceMatcher

from .models import AsrSegment, SubtitleCue, SubtitleMatch


_OVERRIDE_TAG_RE = re.compile(r"\{[^{}]*\}")
_RESET_TAG_RE = re.compile(r"\{\\r[^{}]*\}")
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _source_text(text: str) -> str:
    lines = re.split(r"\\N", text)
    for line in reversed(lines):
        reset = _RESET_TAG_RE.search(line)
        if reset:
            return line[reset.end() :]
    return text


def _normalise(text: str) -> str:
    text = _OVERRIDE_TAG_RE.sub(" ", text)
    tokens = _TOKEN_RE.findall(text.lower())
    return " ".join(tokens)


def _candidate_score(source: str, asr_text: str, cue: SubtitleCue, start: float, end: float) -> float:
    text_score = SequenceMatcher(None, source, _normalise(asr_text)).ratio()
    distance = abs(start - cue.start)
    time_score = max(0.0, 1.0 - min(distance, 60.0) / 60.0)
    return text_score * 0.85 + time_score * 0.15


def match_cues_to_asr(
    cues: Iterable[SubtitleCue],
    asr_segments: Iterable[AsrSegment],
    *,
    max_group_size: int = 4,
    max_time_distance: float = 60.0,
    minimum_score: float = 0.42,
) -> dict[int, SubtitleMatch]:
    ordered_cues = sorted(cues, key=lambda cue: (cue.start, cue.id))
    ordered_asr = sorted(asr_segments, key=lambda segment: (segment.start, segment.id))
    matches: dict[int, SubtitleMatch] = {}

    for cue in ordered_cues:
        source = _normalise(_source_text(cue.text))
        if not source:
            continue

        best: tuple[float, int, int] | None = None
        for start_index in range(len(ordered_asr)):
            first = ordered_asr[start_index]
            if first.start > cue.end + max_time_distance:
                break
            if first.end < cue.start - max_time_distance:
                continue

            combined: list[str] = []
            for end_index in range(start_index, min(len(ordered_asr), start_index + max_group_size)):
                current = ordered_asr[end_index]
                if end_index > start_index and current.start - ordered_asr[end_index - 1].end > 3.0:
                    break
                combined.append(current.text)
                end = current.end
                score = _candidate_score(source, " ".join(combined), cue, first.start, end)
                text_score = SequenceMatcher(
                    None,
                    source,
                    _normalise(" ".join(combined)),
                ).ratio()
                if text_score >= minimum_score and (best is None or score > best[0]):
                    best = (score, start_index, end_index)

        if best is None:
            continue
        score, start_index, end_index = best
        matches[cue.id] = SubtitleMatch(
            subtitle_id=cue.id,
            asr_ids=tuple(segment.id for segment in ordered_asr[start_index : end_index + 1]),
            score=score,
        )

    return matches
