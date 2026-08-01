from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from src.entity.subtitle import (
    DialogueLine,
    NormalizedDialogue,
    NormalizedSubtitle,
    ParsedSubtitle,
    SubtitleSegment,
)
from src.mapper import (
    CorrectionParams,
    DeviationCorrector,
    DeviationWindow,
    detect_pattern,
    median,
)
from src.splitter import SentenceSplitter, get_splitter

_LEAD = 0.2
_END_MARGIN = 0.1
_GAP = 0.1
_OVERLAP_OK = 0.5
_APPEND_MAX_DURATION = 8.0
_LATIN_WRAP = 42
_CJK_WRAP = 22
_DURATION_TOLERANCE = 0.05
_OUT_OF_RANGE_GRACE = 300.0
_CREDIT_RE = re.compile(
    r"翻译|字幕|压制|校对|校译|片源|staff|translated|subtitles|encoding|typeset",
    re.IGNORECASE,
)
_SYMBOL_RE = re.compile(r"^[\s.,!?;:·、，。！？…~\-—()\[\]{}「」『』""'']*$")
_CJK_PUNCT = "、，。！？"
_CJK_PARTICLES = "はがのにをでとも"


@dataclass
class _BEntry:
    source: SubtitleSegment
    start: float
    end: float
    lines: list[DialogueLine]
    action: str = "kept"
    matched_a: int | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _AEntry:
    index: int
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class MappingResult:
    mapped: NormalizedSubtitle
    report: str


def map_transcript(
    a: NormalizedSubtitle,
    b: ParsedSubtitle,
    language: str | None = None,
    english_style: str = "Eng",
    on_log: Callable[[str], None] | None = None,
) -> MappingResult:
    report = _Report()
    splitter = get_splitter(language, _sample_text(a))
    a_entries = [
        _AEntry(dialogue.index, dialogue.start, dialogue.end, _a_text(dialogue, splitter))
        for dialogue in a.dialogue
    ]
    if not a_entries:
        raise RuntimeError("golden transcript is empty; nothing to map against")
    video_end = max(entry.end for entry in a_entries)

    _validate_times(b, video_end)
    b_entries = _filter_entries(b, video_end, report)
    if not b_entries:
        raise RuntimeError(
            f"no valid entries remain after filtering: {b.path}"
        )

    m = len(a_entries)
    n = len(b_entries)
    k = min(m, n)
    report.summary(m, n, k)

    window = DeviationWindow(
        series=[a_entries[i].start - b_entries[i].start for i in range(k)],
        a_start=[a_entries[i].start for i in range(k)],
        a_end=[a_entries[i].end for i in range(k)],
        b_start=[b_entries[i].start for i in range(k)],
        b_end=[b_entries[i].end for i in range(k)],
    )
    corrector, params = detect_pattern(
        window, count_mismatch=m != n, force_overall=k < 5
    )
    report.pattern(params, forced_overall=k < 5)

    pairs = _match(a_entries, b_entries, corrector, k)
    report.matches(len(pairs))

    _correct_times(b_entries, corrector, params)
    _preserve_lead(pairs, params)

    if corrector.match_kind == "overlap":
        _handle_isolated(b_entries, a_entries, pairs, video_end, report)

    _fill_text(b_entries, a_entries, pairs, english_style, splitter)

    if m > n:
        _append_extra(b_entries, a_entries, n, english_style, splitter, report)
    elif m < n:
        _handle_missing(b_entries, a_entries, m, report)

    _validate_output(b_entries, a_entries, pairs, report)

    if on_log is not None:
        on_log(
            f"mapped {m} golden cues onto {n} subtitle entries "
            f"(pattern: {params.pattern})"
        )
    return MappingResult(mapped=_build_output(b, b_entries), report=report.render())


def _a_text(dialogue: NormalizedDialogue, splitter: SentenceSplitter) -> str:
    return splitter.join_text(
        [line.content for line in dialogue.lines if line.content]
    ).strip()


def _sample_text(a: NormalizedSubtitle) -> str:
    parts: list[str] = []
    for dialogue in a.dialogue[:50]:
        for line in dialogue.lines:
            parts.append(line.content)
    return " ".join(parts)


def _validate_times(b: ParsedSubtitle, video_end: float) -> None:
    limit = video_end + _OUT_OF_RANGE_GRACE
    for segment in b.segments:
        if segment.start < 0 or segment.end < 0:
            raise RuntimeError(
                f"invalid negative timestamp in subtitle: #{segment.index} "
                f"{segment.start}..{segment.end}"
            )
        if segment.end > limit:
            raise RuntimeError(
                f"timestamp far beyond video duration: #{segment.index} "
                f"end={segment.end} > {limit}"
            )


def _filter_entries(
    b: ParsedSubtitle, video_end: float, report: _Report
) -> list[_BEntry]:
    kept: list[_BEntry] = []
    for segment in b.segments:
        content = " ".join(line.content for line in segment.lines if line.content)
        if segment.start >= segment.end:
            report.record(
                f"filtered (invalid times): #{segment.index} "
                f"{segment.start}..{segment.end}"
            )
            continue
        if not content.strip():
            report.record(f"filtered (empty content): #{segment.index}")
            continue
        outside = segment.start > video_end or segment.end < 0
        if outside and _CREDIT_RE.search(content):
            report.record(f"filtered (credit outside movie): #{segment.index}")
            continue
        kept.append(
            _BEntry(
                source=segment,
                start=segment.start,
                end=segment.end,
                lines=list(segment.lines),
            )
        )
    return kept


def _match(
    a_entries: list[_AEntry],
    b_entries: list[_BEntry],
    corrector: DeviationCorrector,
    k: int,
) -> list[tuple[_AEntry, _BEntry]]:
    if corrector.match_kind == "sequential":
        return [(a_entries[i], b_entries[i]) for i in range(k)]
    candidates: list[tuple[float, _AEntry, _BEntry]] = []
    for b_entry in b_entries:
        best_ratio, best = 0.0, None
        for a_entry in a_entries:
            ratio = _overlap_ratio(b_entry.start, b_entry.end, a_entry.start, a_entry.end)
            if ratio > best_ratio:
                best_ratio, best = ratio, a_entry
        if best is not None and best_ratio >= _OVERLAP_OK:
            candidates.append((best_ratio, best, b_entry))
    candidates.sort(key=lambda item: item[0], reverse=True)
    pairs: list[tuple[_AEntry, _BEntry]] = []
    taken_a: set[int] = set()
    taken_b: set[int] = set()
    for ratio, a_entry, b_entry in candidates:
        if a_entry.index in taken_a or id(b_entry) in taken_b:
            continue
        pairs.append((a_entry, b_entry))
        taken_a.add(a_entry.index)
        taken_b.add(id(b_entry))
    for b_entry in b_entries:
        if id(b_entry) not in taken_b:
            b_entry.notes.append("unmatched in overlap matching")
    return pairs


def _overlap_ratio(
    b_start: float, b_end: float, a_start: float, a_end: float
) -> float:
    overlap = min(b_end, a_end) - max(b_start, a_start)
    if overlap <= 0:
        return 0.0
    return overlap / max(b_end - b_start, 1e-6)


def _correct_times(
    b_entries: list[_BEntry], corrector: DeviationCorrector, params: CorrectionParams
) -> None:
    if params.pattern in ("discrete", "cumulative"):
        return
    times = corrector.correct_times(
        [(entry.start, entry.end) for entry in b_entries], params
    )
    for entry, (start, end) in zip(b_entries, times, strict=True):
        entry.start, entry.end = start, end


def _preserve_lead(
    pairs: list[tuple[_AEntry, _BEntry]], params: CorrectionParams
) -> None:
    if params.pattern not in ("overall", "segmented"):
        return
    for a_entry, b_entry in pairs:
        if b_entry.start > a_entry.start:
            b_entry.start = max(0.0, a_entry.start - _LEAD)


def _handle_isolated(
    b_entries: list[_BEntry],
    a_entries: list[_AEntry],
    pairs: list[tuple[_AEntry, _BEntry]],
    video_end: float,
    report: _Report,
) -> None:
    paired = {id(b_entry) for _, b_entry in pairs}
    for b_entry in b_entries:
        if id(b_entry) in paired:
            continue
        content = " ".join(line.content for line in b_entry.lines)
        if _SYMBOL_RE.match(content):
            b_entry.action = "deleted"
            report.record(
                f"isolated deleted (symbols only): #{b_entry.source.index}"
            )
            continue
        if (b_entry.start > video_end or b_entry.end < 0) and _CREDIT_RE.search(
            content
        ):
            report.record(
                f"isolated kept unchanged (credit outside movie): "
                f"#{b_entry.source.index}"
            )
            continue
        closest = max(
            a_entries,
            key=lambda a_entry: _overlap_ratio(
                b_entry.start, b_entry.end, a_entry.start, a_entry.end
            ),
        )
        b_entry.start = max(0.0, closest.start - _LEAD)
        b_entry.end = closest.end + _END_MARGIN
        b_entry.matched_a = closest.index
        b_entry.action = "overwritten"
        report.record(
            f"isolated overwritten with nearest cue: #{b_entry.source.index} "
            f"<- A#{closest.index}"
        )


def _fill_text(
    b_entries: list[_BEntry],
    a_entries: list[_AEntry],
    pairs: list[tuple[_AEntry, _BEntry]],
    english_style: str,
    splitter: SentenceSplitter,
) -> None:
    for a_entry, b_entry in pairs:
        b_entry.matched_a = a_entry.index
    by_index = {entry.index: entry for entry in a_entries}
    for b_entry in b_entries:
        if b_entry.matched_a is None:
            continue
        a_entry = by_index.get(b_entry.matched_a)
        if a_entry is None:
            continue
        wrapped = _wrap_text(a_entry.text, splitter)
        new_lines: list[DialogueLine] = []
        inserted = False
        for line in b_entry.lines:
            if line.style == english_style:
                if not inserted:
                    new_lines.extend(
                        DialogueLine(style=english_style, content=part)
                        for part in wrapped
                    )
                    inserted = True
            else:
                new_lines.append(line)
        if not inserted:
            new_lines.extend(
                DialogueLine(style=english_style, content=part) for part in wrapped
            )
        b_entry.lines = new_lines


def _wrap_text(text: str, splitter: SentenceSplitter) -> list[str]:
    cjk = splitter.family == "cjk"
    limit = _CJK_WRAP if cjk else _LATIN_WRAP
    if len(text) <= limit:
        return [text]
    lines: list[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = _cjk_cut(window) if cjk else window.rfind(" ")
        if cut <= 0:
            cut = limit
        lines.append(remaining[:cut])
        remaining = remaining[cut:].strip()
        if not remaining:
            break
    if remaining:
        lines.append(remaining)
    return lines


def _cjk_cut(window: str) -> int:
    best = max([window.rfind(char) for char in _CJK_PUNCT] + [0])
    if best > 0:
        return best + 1
    best = max([window.rfind(char) for char in _CJK_PARTICLES] + [0])
    return best + 1 if best > 0 else 0


def _append_extra(
    b_entries: list[_BEntry],
    a_entries: list[_AEntry],
    n: int,
    english_style: str,
    splitter: SentenceSplitter,
    report: _Report,
) -> None:
    last = b_entries[-1]
    extras = a_entries[n:]
    pending = [
        line.content for line in last.lines if line.style == english_style
    ]
    target_end = last.start + _APPEND_MAX_DURATION
    remainder: list[_AEntry] = []
    for extra in extras:
        if extra.end > target_end:
            remainder.append(extra)
        else:
            pending.append(extra.text)
            last.end = extra.end
            report.record(f"appended A#{extra.index} text to B#{last.source.index}")
    _set_english_lines(last, splitter.join_text(pending).strip(), english_style, splitter)
    for extra in remainder:
        new_entry = _BEntry(
            source=last.source,
            start=max(0.0, extra.start - _LEAD),
            end=extra.end + _END_MARGIN,
            lines=[],
            action="appended",
            matched_a=extra.index,
        )
        _set_english_lines(new_entry, extra.text, english_style, splitter)
        b_entries.append(new_entry)
        report.record(
            f"extra A#{extra.index} appended as new entry: "
            f"{new_entry.start}..{new_entry.end}"
        )


def _set_english_lines(
    entry: _BEntry, text: str, english_style: str, splitter: SentenceSplitter
) -> None:
    if not text:
        return
    entry.lines = [
        line
        for line in entry.lines
        if line.style != english_style
    ] + [
        DialogueLine(style=english_style, content=part)
        for part in _wrap_text(text, splitter)
    ]


def _handle_missing(
    b_entries: list[_BEntry],
    a_entries: list[_AEntry],
    m: int,
    report: _Report,
) -> None:
    for b_entry in b_entries[m:]:
        if b_entry.matched_a is not None:
            continue
        content = " ".join(line.content for line in b_entry.lines)
        if _SYMBOL_RE.match(content):
            b_entry.action = "deleted"
            report.record(
                f"missing-text deleted (symbols only): #{b_entry.source.index}"
            )
        elif _CREDIT_RE.search(content):
            report.record(
                f"missing-text kept unchanged (credit info): #{b_entry.source.index}"
            )
        else:
            report.warning(
                f"missing text for valid speech, original kept: "
                f"#{b_entry.source.index} {content[:30]!r}"
            )


def _validate_output(
    b_entries: list[_BEntry],
    a_entries: list[_AEntry],
    pairs: list[tuple[_AEntry, _BEntry]],
    report: _Report,
) -> None:
    kept = [entry for entry in b_entries if entry.action != "deleted"]

    fixed = 0
    for index in range(len(kept) - 1):
        if kept[index].end > kept[index + 1].start:
            new_end = kept[index + 1].start - _GAP
            if new_end >= kept[index].start:
                kept[index].end = new_end
            else:
                kept[index + 1].start = kept[index].end + _GAP
            fixed += 1
    report.validation(
        "continuity",
        "PASS" if fixed == 0 else f"FIXED ({fixed} overlaps resolved)",
    )

    overwritten = 0
    for a_entry, b_entry in pairs:
        if b_entry.action == "deleted":
            continue
        if _overlap_ratio(b_entry.start, b_entry.end, a_entry.start, a_entry.end) < _OVERLAP_OK:
            b_entry.start = max(0.0, a_entry.start - _LEAD)
            b_entry.end = a_entry.end + _END_MARGIN
            overwritten += 1
            report.record(
                f"severe deviation fixed by overwrite: "
                f"B#{b_entry.source.index} <- A#{a_entry.index}"
            )
    report.validation(
        "overlap",
        "PASS" if overwritten == 0 else f"FIXED ({overwritten} cues overwritten)",
    )

    if kept:
        a_total = a_entries[-1].end - a_entries[0].start
        b_total = kept[-1].end - kept[0].start
        if a_total > 0 and abs(b_total - a_total) / a_total > _DURATION_TOLERANCE:
            _linear_retry(kept, a_entries, report)
        else:
            report.validation("total duration", "PASS")

    blank_indices = {
        b_entry.source.index
        for b_entry in kept
        if b_entry.matched_a is not None
        and not " ".join(
            line.content for line in b_entry.lines if line.content.strip()
        )
    }
    for b_entry in kept:
        if b_entry.source.index not in blank_indices:
            continue
        neighbor = next(
            (
                other
                for other in kept
                if other.source.index not in blank_indices
                and other.matched_a is not None
            ),
            None,
        )
        if neighbor is not None:
            b_entry.lines = neighbor.lines
            report.warning(
                f"blank subtitle filled from neighbor: #{b_entry.source.index}"
            )
    report.validation("blank text", "PASS" if not blank_indices else f"FILLED ({len(blank_indices)})")


def _linear_retry(
    entries: list[_BEntry], a_entries: list[_AEntry], report: _Report
) -> None:
    first_a, last_a = a_entries[0], a_entries[-1]
    if not entries:
        return
    first_b, last_b = entries[0], entries[-1]
    span_b = last_b.end - first_b.start
    if span_b <= 0:
        report.validation("total duration", "FAIL (cannot rescale)")
        return
    ratio = (last_a.end - first_a.start) / span_b
    offset = first_a.start - first_b.start * ratio
    for entry in entries:
        entry.start = entry.start * ratio + offset
        entry.end = entry.end * ratio + offset
    report.validation(
        "total duration",
        f"FIXED (re-scaled with R={ratio:.4f}, offset={offset:.3f})",
    )


def _build_output(b: ParsedSubtitle, b_entries: list[_BEntry]) -> NormalizedSubtitle:
    dialogue: list[NormalizedDialogue] = []
    index = 1
    for entry in b_entries:
        if entry.action == "deleted":
            continue
        dialogue.append(
            NormalizedDialogue(
                index=index,
                start=round(entry.start, 3),
                end=round(entry.end, 3),
                lines=entry.lines,
            )
        )
        index += 1
    return NormalizedSubtitle(
        path=b.path, format=b.format, styles=b.styles, dialogue=dialogue
    )


class _Report:
    def __init__(self) -> None:
        self._lines: list[str] = []

    def summary(self, m: int, n: int, k: int) -> None:
        self._lines.append(f"Input summary: A(M)={m}, B(N')={n}, samples K={k}")

    def pattern(self, params: CorrectionParams, forced_overall: bool) -> None:
        suffix = " (forced: K<5)" if forced_overall else ""
        self._lines.append(f"Deviation pattern: {params.pattern}{suffix}")
        if params.pattern == "overall":
            self._lines.append(f"  delta = {params.delta:+.3f}s")
        elif params.pattern == "linear":
            self._lines.append(
                f"  ratio = {params.ratio:.6f}, offset = {params.offset:+.3f}s"
            )
        elif params.pattern == "segmented":
            for index, boundary in enumerate(params.breakpoints):
                self._lines.append(
                    f"  segment {index}: entries <#{boundary}, "
                    f"delta = {params.segment_deltas[index]:+.3f}s"
                )
            self._lines.append(
                f"  segment {len(params.breakpoints)}: last, "
                f"delta = {params.segment_deltas[-1]:+.3f}s"
            )

    def matches(self, count: int) -> None:
        self._lines.append(f"Matched pairs: {count}")

    def record(self, message: str) -> None:
        self._lines.append(f"  - {message}")

    def warning(self, message: str) -> None:
        self._lines.append(f"  ! WARNING: {message}")

    def validation(self, name: str, result: str) -> None:
        self._lines.append(f"  {name}: {result}")

    def render(self) -> str:
        return "\n".join(self._lines)
