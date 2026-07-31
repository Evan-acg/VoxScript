from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any

import pysubs2

from .ass_io import AssTagError, LoadedSubtitle, mask_ass_tags, restore_ass_tags
from .models import ApplyReport, AsrSegment, ReviewOperation


AUTOMATIC_ACTIONS = frozenset({"keep", "revise", "insert"})


def _next_dialogue_start(document: LoadedSubtitle, event: Any) -> float | None:
    starts = [
        candidate.start / 1000
        for candidate in document.events
        if getattr(candidate, "type", "Dialogue") == "Dialogue"
        and candidate is not event
        and candidate.start > event.start
    ]
    return min(starts) if starts else None


def _next_dialogue_start_after(document: LoadedSubtitle, start_ms: int) -> float | None:
    starts = [
        event.start / 1000
        for event in document.events
        if getattr(event, "type", "Dialogue") == "Dialogue" and event.start > start_ms
    ]
    return min(starts) if starts else None


def _asr_window(
    operation: ReviewOperation,
    asr_by_id: dict[int, AsrSegment],
    *,
    start_padding: float,
    end_padding: float,
    next_start: float | None,
    minimum_gap: float,
    max_end: float | None,
) -> tuple[int, int] | None:
    if not operation.asr_ids or any(item not in asr_by_id for item in operation.asr_ids):
        return None

    segments = [asr_by_id[item] for item in operation.asr_ids]
    start = max(0.0, min(item.start for item in segments) - start_padding)
    end = max(item.end for item in segments) + end_padding
    if max_end is not None:
        end = min(end, max_end)
    if next_start is not None:
        end = min(end, next_start - minimum_gap)
    if end <= start:
        return None
    return round(start * 1000), round(end * 1000)


def _revised_text(original: str, proposed: str) -> str:
    _, tokens = mask_ass_tags(original)
    result = restore_ass_tags(proposed, tokens)
    if not result.strip():
        raise AssTagError("revised subtitle text is empty")
    return result


def _nearest_template(document: LoadedSubtitle, start_ms: int) -> Any | None:
    dialogue_events = [
        event
        for event in document.events
        if getattr(event, "type", "Dialogue") == "Dialogue"
    ]
    if not dialogue_events:
        return None
    return min(dialogue_events, key=lambda event: abs(event.start - start_ms))


def _insert_event(
    document: LoadedSubtitle,
    operation: ReviewOperation,
    start_ms: int,
    end_ms: int,
) -> Any:
    template = _nearest_template(document, start_ms)
    if template is None:
        event = pysubs2.SSAEvent(start=start_ms, end=end_ms, text=operation.text)
    else:
        event = copy.deepcopy(template)
        event.start = start_ms
        event.end = end_ms
        event.text = operation.text
    document.add_event(event)
    return event


def apply_operations(
    document: LoadedSubtitle,
    operations: Iterable[ReviewOperation],
    asr_segments: Iterable[AsrSegment],
    *,
    start_padding: float = 0.10,
    end_padding: float = 0.20,
    minimum_gap: float = 0.05,
    max_end: float | None = None,
    clamp_to_next: bool = True,
) -> ApplyReport:
    asr_by_id = {segment.id: segment for segment in asr_segments}
    applied: list[int | str] = []
    unresolved: list[ReviewOperation] = []

    for operation in operations:
        if operation.action not in AUTOMATIC_ACTIONS:
            unresolved.append(operation)
            continue

        if operation.action == "insert":
            if operation.subtitle_ids:
                unresolved.append(operation)
                continue
            insert_next_start = None
            if operation.asr_ids and all(item in asr_by_id for item in operation.asr_ids):
                insert_next_start = _next_dialogue_start_after(
                    document,
                    round(max(asr_by_id[item].start for item in operation.asr_ids) * 1000),
                )
            window = _asr_window(
                operation,
                asr_by_id,
                start_padding=start_padding,
                end_padding=end_padding,
                next_start=insert_next_start if clamp_to_next else None,
                minimum_gap=minimum_gap,
                max_end=max_end,
            )
            if window is None or not operation.text.strip():
                unresolved.append(operation)
                continue
            try:
                text = restore_ass_tags(operation.text, ())
            except AssTagError:
                unresolved.append(operation)
                continue
            _insert_event(
                document,
                ReviewOperation(
                    action=operation.action,
                    subtitle_ids=operation.subtitle_ids,
                    asr_ids=operation.asr_ids,
                    text=text,
                    reason=operation.reason,
                ),
                *window,
            )
            applied.append(f"insert:{','.join(str(item) for item in operation.asr_ids)}")
            continue

        if len(operation.subtitle_ids) != 1:
            unresolved.append(operation)
            continue

        subtitle_id = operation.subtitle_ids[0]
        event = document.events_by_id.get(subtitle_id)
        if event is None:
            unresolved.append(operation)
            continue

        window = _asr_window(
            operation,
            asr_by_id,
            start_padding=start_padding,
            end_padding=end_padding,
            next_start=_next_dialogue_start(document, event) if clamp_to_next else None,
            minimum_gap=minimum_gap,
            max_end=max_end,
        )
        if window is None:
            unresolved.append(operation)
            continue

        if operation.action == "revise":
            try:
                event.text = _revised_text(event.text, operation.text)
            except AssTagError:
                unresolved.append(operation)
                continue

        event.start, event.end = window
        applied.append(subtitle_id)

    return ApplyReport(applied=applied, unresolved=unresolved)
