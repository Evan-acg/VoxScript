from __future__ import annotations

import re

from .models import AsrSegment


_SENTENCE_RE = re.compile(r"[^.!?。！？]+[.!?。！？]+|[^.!?。！？]+$", re.UNICODE)


def split_asr_segment(segment: AsrSegment) -> list[AsrSegment]:
    parts = [part.strip() for part in _SENTENCE_RE.findall(segment.text) if part.strip()]
    if len(parts) <= 1 or segment.end <= segment.start:
        return [segment]

    weights = [max(1, len(re.sub(r"\s+", "", part))) for part in parts]
    total_weight = sum(weights)
    duration = segment.end - segment.start
    result: list[AsrSegment] = []
    current = segment.start
    for index, (part, weight) in enumerate(zip(parts, weights, strict=True)):
        if index == len(parts) - 1:
            end = segment.end
        else:
            end = current + duration * weight / total_weight
        result.append(
            AsrSegment(
                id=segment.id,
                start=current,
                end=end,
                text=part,
                chunk_id=segment.chunk_id,
            )
        )
        current = end
    return result
