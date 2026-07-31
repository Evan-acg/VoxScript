from __future__ import annotations

import math

from .models import Chunk


def build_chunks(
    duration: float,
    *,
    chunk_minutes: float = 10,
    context_seconds: float = 10,
) -> list[Chunk]:
    if not math.isfinite(duration) or duration <= 0:
        return []
    if chunk_minutes <= 0:
        raise ValueError("chunk_minutes must be greater than zero")
    if context_seconds < 0:
        raise ValueError("context_seconds cannot be negative")

    chunk_seconds = chunk_minutes * 60
    chunks: list[Chunk] = []
    body_start = 0.0
    chunk_id = 0
    while body_start < duration:
        body_end = min(duration, body_start + chunk_seconds)
        chunks.append(
            Chunk(
                id=chunk_id,
                body_start=body_start,
                body_end=body_end,
                context_start=max(0.0, body_start - context_seconds),
                context_end=min(duration, body_end + context_seconds),
            )
        )
        body_start = body_end
        chunk_id += 1
    return chunks


def belongs_to_body(start: float, chunk: Chunk) -> bool:
    return chunk.body_start <= start < chunk.body_end
