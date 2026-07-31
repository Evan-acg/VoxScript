from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from ..progress import ProgressCallback, ProgressEvent, null_callback
from .ass_io import LoadedSubtitle, load_ass, mask_ass_tags, save_ass
from .chunks import belongs_to_body, build_chunks
from .matcher import match_cues_to_asr
from .models import AsrSegment, Chunk, ReviewOperation, SubtitleMatch
from .operations import apply_operations
from .segmenter import split_asr_segment


class RepairError(Exception):
    pass


class AudioChunkExtractor(Protocol):
    def get_duration(self, input_path: str) -> float: ...

    def extract(
        self,
        input_path: str,
        output_path: str,
        *,
        stream_index: int | None,
        start: float,
        end: float,
        force: bool,
        on_progress: ProgressCallback,
    ) -> Any: ...


class AsrRunner(Protocol):
    def transcribe(
        self,
        audio_path: str,
        *,
        offset: float,
        chunk_id: int | None,
        model_name: str,
        language: str | None,
        device: str,
        on_progress: ProgressCallback,
    ) -> list[AsrSegment]: ...


class ProposalRunner(Protocol):
    def propose(
        self,
        *,
        chunk_id: int,
        subtitle_entries: list[dict[str, Any]],
        asr_entries: list[dict[str, Any]],
        body_subtitle_ids: set[int],
        body_asr_ids: set[int],
        source_language: str | None,
        target_language: str,
        on_progress: ProgressCallback,
    ) -> list[ReviewOperation]: ...


@dataclass(frozen=True)
class RepairOptions:
    chunk_minutes: float = 10
    context_seconds: float = 10
    model_name: str = "base"
    language: str | None = None
    target_language: str = "auto"
    device: str = "cuda"
    vad_method: str = "silero"
    batch_size: int | None = None
    asr_chunk_seconds: int = 5
    align: bool = False
    track_index: int | None = None
    force: bool = False
    work_dir: Path | None = None
    keep_artifacts: bool = False


@dataclass(frozen=True)
class RepairResult:
    output_path: Path
    asr_path: Path | None
    review_path: Path | None
    preview_path: Path | None
    partial: bool = False


def _operation_dict(operation: ReviewOperation) -> dict[str, Any]:
    return {
        "action": operation.action,
        "subtitle_ids": list(operation.subtitle_ids),
        "asr_ids": list(operation.asr_ids),
        "text": operation.text,
        "reason": operation.reason,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_reports(
    asr_path: Path,
    review_path: Path,
    asr_segments: list[AsrSegment],
    applied: list[int | str],
    operations: list[ReviewOperation],
    unresolved: list[ReviewOperation],
    errors: list[dict[str, Any]],
) -> None:
    _write_json(
        asr_path,
        {
            "segments": [asdict(segment) for segment in asr_segments],
        },
    )
    _write_json(
        review_path,
        {
            "applied": applied,
            "operations": [_operation_dict(operation) for operation in operations],
            "unresolved": [_operation_dict(operation) for operation in unresolved],
            "errors": errors,
        },
    )


def _cue_entries(document: LoadedSubtitle, chunk: Chunk) -> tuple[list[dict[str, Any]], set[int]]:
    entries: list[dict[str, Any]] = []
    body_ids: set[int] = set()
    for cue_id, cue in document.cues.items():
        if not chunk.context_start <= cue.start < chunk.context_end:
            continue
        event = document.events_by_id[cue_id]
        masked_text, _ = mask_ass_tags(event.text)
        is_body = cue_id > 0 and belongs_to_body(cue.start, chunk)
        if is_body:
            body_ids.add(cue_id)
        entries.append(
            {
                "id": cue_id,
                "start": event.start / 1000,
                "end": event.end / 1000,
                "text": masked_text,
                "scope": "body" if is_body else "context",
                "style": event.style,
            }
        )
    return entries, body_ids


def _normalise_asr_segments(
    segments: Iterable[AsrSegment],
    *,
    chunk_id: int | None,
    next_id: int,
) -> tuple[list[AsrSegment], int]:
    normalised: list[AsrSegment] = []
    for segment in segments:
        if segment.end <= segment.start or not segment.text.strip():
            continue
        normalised.append(
            AsrSegment(
                id=next_id,
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
                chunk_id=chunk_id,
            )
        )
        next_id += 1
    return normalised, next_id


def _segments_for_chunk(segments: Iterable[AsrSegment], chunk: Chunk) -> list[AsrSegment]:
    return [
        segment
        for segment in segments
        if chunk.context_start <= segment.start < chunk.context_end
    ]


def _set_asr_owners(segments: Iterable[AsrSegment], chunks: list[Chunk]) -> list[AsrSegment]:
    owned: list[AsrSegment] = []
    for segment in segments:
        owner = next(
            (chunk.id for chunk in chunks if belongs_to_body(segment.start, chunk)),
            None,
        )
        owned.append(
            AsrSegment(
                id=segment.id,
                start=segment.start,
                end=segment.end,
                text=segment.text,
                chunk_id=owner,
            )
        )
    return owned


def _timing_operations(
    matches: dict[int, SubtitleMatch],
    operations: Iterable[ReviewOperation],
) -> list[ReviewOperation]:
    blocked = {
        subtitle_id
        for operation in operations
        if operation.action in {"delete", "review", "split", "merge"}
        for subtitle_id in operation.subtitle_ids
    }
    return [
        ReviewOperation(
            action="keep",
            subtitle_ids=(subtitle_id,),
            asr_ids=match.asr_ids,
            text="",
            reason=f"deterministic ASR match score={match.score:.3f}",
        )
        for subtitle_id, match in matches.items()
        if subtitle_id not in blocked
    ]


def _expand_operation_evidence(
    operations: Iterable[ReviewOperation],
    matches: dict[int, SubtitleMatch],
) -> list[ReviewOperation]:
    expanded: list[ReviewOperation] = []
    for operation in operations:
        if operation.action in {"keep", "revise"} and len(operation.subtitle_ids) == 1:
            match = matches.get(operation.subtitle_ids[0])
            if match is not None:
                expanded.append(replace(operation, asr_ids=match.asr_ids))
                continue
        expanded.append(operation)
    return expanded


def _unique_values(values: Iterable[int | str]) -> list[int | str]:
    return list(dict.fromkeys(values))


def _asr_entries(segments: Iterable[AsrSegment], chunk: Chunk) -> tuple[list[dict[str, Any]], set[int]]:
    entries: list[dict[str, Any]] = []
    body_ids: set[int] = set()
    for segment in segments:
        is_body = belongs_to_body(segment.start, chunk)
        if is_body:
            body_ids.add(segment.id)
        entries.append(
            {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "scope": "body" if is_body else "context",
            }
        )
    return entries, body_ids


def _validate_operations(
    operations: Iterable[ReviewOperation],
    *,
    subtitle_ids: set[int],
    asr_ids: set[int],
    body_subtitle_ids: set[int],
    body_asr_ids: set[int],
) -> None:
    used_subtitle_ids: set[int] = set()
    used_insert_asr_ids: set[int] = set()
    for operation in operations:
        if any(item not in subtitle_ids for item in operation.subtitle_ids):
            raise ValueError("proposal contains an unknown subtitle id")
        if any(item not in asr_ids for item in operation.asr_ids):
            raise ValueError("proposal contains an unknown ASR id")
        if any(item not in body_subtitle_ids for item in operation.subtitle_ids):
            raise ValueError("proposal modifies a subtitle outside chunk body")
        if any(item not in body_asr_ids for item in operation.asr_ids):
            raise ValueError("proposal uses ASR outside chunk body")
        if used_subtitle_ids.intersection(operation.subtitle_ids):
            raise ValueError("proposal contains duplicate subtitle ids")
        if operation.action == "insert":
            duplicate_insert_asr = used_insert_asr_ids.intersection(operation.asr_ids)
            if duplicate_insert_asr:
                raise ValueError("proposal contains duplicate insert ASR ids")
        used_subtitle_ids.update(operation.subtitle_ids)
        if operation.action == "insert":
            used_insert_asr_ids.update(operation.asr_ids)


def _chunk_payload(
    chunk: Chunk,
    document: LoadedSubtitle,
    asr_segments: list[AsrSegment],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[int], set[int]]:
    subtitle_entries, body_subtitle_ids = _cue_entries(document, chunk)
    asr_entries, body_asr_ids = _asr_entries(asr_segments, chunk)
    return subtitle_entries, asr_entries, body_subtitle_ids, body_asr_ids


def _same_file(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except FileNotFoundError:
        return left.resolve() == right.resolve()


def run_repair(
    video_path: str | Path,
    subtitle_path: str | Path,
    output_path: str | Path,
    *,
    audio: AudioChunkExtractor,
    asr: AsrRunner,
    llm: ProposalRunner,
    options: RepairOptions = RepairOptions(),
    on_progress: ProgressCallback = null_callback,
) -> RepairResult:
    video = Path(video_path)
    original = Path(subtitle_path)
    output = Path(output_path)
    if _same_file(original, output):
        raise RepairError("output must not overwrite the original ASS file")
    work_dir = options.work_dir or output.parent / f".{output.stem}.voxscript"
    work_dir.mkdir(parents=True, exist_ok=True)
    asr_path = work_dir / "asr.json"
    review_path = work_dir / "review.json"
    preview_path = work_dir / "preview.ass"

    document = load_ass(original)
    duration = audio.get_duration(str(video))
    if not math.isfinite(duration) or duration <= 0:
        raise RepairError(f"media duration is not a positive finite number: {duration}")
    chunks = build_chunks(
        duration,
        chunk_minutes=options.chunk_minutes,
        context_seconds=options.context_seconds,
    )
    if not chunks:
        raise RepairError("media has no processable chunks")

    errors: list[dict[str, Any]] = []
    all_operations: list[ReviewOperation] = []
    detected_language: str | None = options.language
    next_asr_id = 1

    with tempfile.TemporaryDirectory(prefix="voxscript_repair_") as temp_dir:
        audio_path = Path(temp_dir) / "full-audio.wav"
        try:
            audio.extract(
                str(video),
                str(audio_path),
                stream_index=options.track_index,
                start=0.0,
                end=duration,
                force=options.force,
                on_progress=on_progress,
            )
            raw_segments = asr.transcribe(
                str(audio_path),
                offset=0.0,
                chunk_id=None,
                model_name=options.model_name,
                language=options.language,
                device=options.device,
                vad_method=options.vad_method,
                batch_size=options.batch_size,
                chunk_size=options.asr_chunk_seconds,
                align=options.align,
                on_progress=on_progress,
            )
            detected_language = getattr(asr, "last_language", None) or detected_language
        except Exception as error:
            errors.append({"stage": "asr", "message": str(error)})
            _write_reports(asr_path, review_path, [], [], [], [], errors)
            raise RepairError(f"full ASR failed; see {review_path}") from error

    sentence_segments = [
        part
        for segment in raw_segments
        for part in split_asr_segment(segment)
    ]
    all_asr, _ = _normalise_asr_segments(
        sentence_segments,
        chunk_id=None,
        next_id=next_asr_id,
    )
    all_asr = _set_asr_owners(all_asr, chunks)

    for chunk in chunks:
        on_progress(
            ProgressEvent(
                "repair",
                chunk.id + 1,
                len(chunks),
                f"Processing LLM chunk {chunk.id + 1}/{len(chunks)}",
            )
        )
        chunk_asr = _segments_for_chunk(all_asr, chunk)
        subtitle_entries, asr_entries, body_subtitle_ids, body_asr_ids = _chunk_payload(
            chunk,
            document,
            chunk_asr,
        )
        if not body_subtitle_ids and not body_asr_ids:
            continue
        try:
            operations = llm.propose(
                chunk_id=chunk.id,
                subtitle_entries=subtitle_entries,
                asr_entries=asr_entries,
                body_subtitle_ids=body_subtitle_ids,
                body_asr_ids=body_asr_ids,
                source_language=detected_language,
                target_language=options.target_language,
                on_progress=on_progress,
            )
            subtitle_ids = {entry["id"] for entry in subtitle_entries}
            asr_ids = {entry["id"] for entry in asr_entries}
            _validate_operations(
                operations,
                subtitle_ids=subtitle_ids,
                asr_ids=asr_ids,
                body_subtitle_ids=body_subtitle_ids,
                body_asr_ids=body_asr_ids,
            )
            all_operations.extend(operations)
        except Exception as error:
            errors.append(
                {
                    "chunk_id": chunk.id,
                    "stage": "llm",
                    "message": str(error),
                }
            )

    matches = match_cues_to_asr(document.cues.values(), all_asr)
    all_operations = _expand_operation_evidence(all_operations, matches)
    timing_operations = _timing_operations(matches, all_operations)
    report_operations = timing_operations + all_operations
    apply_report = apply_operations(
        document,
        report_operations,
        all_asr,
        max_end=duration,
        clamp_to_next=False,
    )
    all_applied = _unique_values(apply_report.applied)
    all_unresolved = apply_report.unresolved

    save_ass(document, preview_path)
    _write_reports(
        asr_path,
        review_path,
        all_asr,
        all_applied,
        report_operations,
        all_unresolved,
        errors,
    )

    load_ass(preview_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(preview_path, output)

    retain_artifacts = options.keep_artifacts or bool(errors or all_unresolved)
    partial = bool(errors or all_unresolved)
    if not retain_artifacts and options.work_dir is None:
        shutil.rmtree(work_dir, ignore_errors=True)
        return RepairResult(output, None, None, None, partial=False)
    return RepairResult(output, asr_path, review_path, preview_path, partial=partial)
