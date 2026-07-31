from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .models import ReviewOperation


class ProposalError(ValueError):
    pass


ALLOWED_ACTIONS = frozenset({"keep", "revise", "insert", "delete", "review", "split", "merge"})


def _id_list(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ProposalError(f"{field} must be a list of integers")
    return tuple(value)


def _check_ids(
    ids: Iterable[int],
    known: set[int],
    body: set[int],
    label: str,
) -> None:
    for item in ids:
        if item not in known:
            raise ProposalError(f"unknown {label} id: {item}")
        if item not in body:
            raise ProposalError(f"{label} id {item} is outside chunk body")


def parse_operations(
    payload: Any,
    *,
    subtitle_ids: set[int],
    asr_ids: set[int],
    body_subtitle_ids: set[int],
    body_asr_ids: set[int],
) -> list[ReviewOperation]:
    if not isinstance(payload, dict) or not isinstance(payload.get("operations"), list):
        raise ProposalError("response must contain an operations list")

    operations: list[ReviewOperation] = []
    used_subtitle_ids: set[int] = set()
    used_insert_asr_ids: set[int] = set()

    for raw in payload["operations"]:
        if not isinstance(raw, dict):
            raise ProposalError("each operation must be an object")
        action = raw.get("action")
        if not isinstance(action, str) or action not in ALLOWED_ACTIONS:
            raise ProposalError(f"unsupported action: {action}")

        current_subtitle_ids = _id_list(raw.get("subtitle_ids"), "subtitle_ids")
        current_asr_ids = _id_list(raw.get("asr_ids"), "asr_ids")
        if len(set(current_subtitle_ids)) != len(current_subtitle_ids):
            raise ProposalError("duplicate subtitle id in operation")
        if len(set(current_asr_ids)) != len(current_asr_ids):
            raise ProposalError("duplicate ASR id in operation")
        _check_ids(current_subtitle_ids, subtitle_ids, body_subtitle_ids, "subtitle")
        _check_ids(current_asr_ids, asr_ids, body_asr_ids, "ASR")

        duplicate_subtitles = used_subtitle_ids.intersection(current_subtitle_ids)
        if duplicate_subtitles:
            duplicate = min(duplicate_subtitles)
            raise ProposalError(f"duplicate subtitle id: {duplicate}")
        if action == "insert" and current_subtitle_ids:
            raise ProposalError("insert cannot contain subtitle ids")
        if action == "insert":
            duplicate_insert_asr = used_insert_asr_ids.intersection(current_asr_ids)
            if duplicate_insert_asr:
                duplicate = min(duplicate_insert_asr)
                raise ProposalError(f"duplicate insert ASR id: {duplicate}")
        if action in {"keep", "revise", "delete", "review", "split", "merge"} and not current_subtitle_ids:
            raise ProposalError(f"{action} requires subtitle ids")
        if action in {"keep", "revise", "insert"} and not current_asr_ids:
            raise ProposalError(f"{action} requires ASR ids")

        text = raw.get("text", "")
        reason = raw.get("reason", "")
        if not isinstance(text, str) or not isinstance(reason, str):
            raise ProposalError("text and reason must be strings")
        if action in {"revise", "insert"} and not text.strip():
            raise ProposalError(f"{action} requires non-empty text")

        operations.append(
            ReviewOperation(
                action=action,
                subtitle_ids=current_subtitle_ids,
                asr_ids=current_asr_ids,
                text=text,
                reason=reason,
            )
        )
        used_subtitle_ids.update(current_subtitle_ids)
        if action == "insert":
            used_insert_asr_ids.update(current_asr_ids)

    return operations
