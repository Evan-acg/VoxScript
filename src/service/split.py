from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.entity.subtitle import DialogueLine, NormalizedDialogue, NormalizedSubtitle
from src.splitter import SentenceSplitter, get_splitter

MIN_DURATION = 0.6
MAX_DURATION = 8.0
MERGE_THRESHOLD = 1.0
MERGE_MAX = 3.0
ROUND = 3
_EPSILON = 1e-9
_DUP_WINDOW = 1.5


@dataclass(frozen=True)
class _Cue:
    text: str
    start: float
    end: float


def split_transcript_file(
    input_path: Path,
    output_path: Path | None = None,
    language: str | None = None,
    on_log: Callable[[str], None] | None = None,
) -> NormalizedSubtitle:
    normalized = NormalizedSubtitle.model_validate_json(
        input_path.read_text(encoding="utf-8")
    )
    split = split_transcript(normalized, language=language, on_log=on_log)
    if output_path is not None:
        text = split.model_dump_json(indent=2, ensure_ascii=False) + "\n"
        output_path.write_text(text, encoding="utf-8")
    return split


def split_transcript(
    normalized: NormalizedSubtitle,
    language: str | None = None,
    on_log: Callable[[str], None] | None = None,
) -> NormalizedSubtitle:
    splitter = get_splitter(language, _sample_text(normalized))
    output: list[NormalizedDialogue] = []
    index = 1
    for dialogue in normalized.dialogue:
        sentences = _split_dialogue(dialogue, splitter)
        if len(sentences) <= 1:
            output.append(_single(dialogue, sentences, index))
            index += 1
        else:
            cues = _merge_short(
                _allocate(dialogue.start, dialogue.end, sentences), splitter
            )
            for cue in cues:
                output.append(
                    NormalizedDialogue(
                        index=index,
                        start=round(cue.start, ROUND),
                        end=round(cue.end, ROUND),
                        lines=[DialogueLine(style="Default", content=cue.text.strip())],
                    )
                )
                index += 1
    if on_log is not None:
        on_log(f"split into {len(output)} cues")
    output = _dedup_clusters(output)
    if on_log is not None:
        on_log(f"deduplicated whisper repeat clusters: {len(output)} cues")
    return normalized.model_copy(update={"dialogue": output})


def _dedup_clusters(output: list[NormalizedDialogue]) -> list[NormalizedDialogue]:
    seen: dict[str, float] = {}
    kept: list[NormalizedDialogue] = []
    for dialogue in output:
        text = _dialogue_text(dialogue)
        if not text:
            kept.append(dialogue)
            continue
        last = seen.get(text)
        if last is not None and abs(dialogue.start - last) <= _DUP_WINDOW:
            continue
        seen[text] = dialogue.start
        kept.append(dialogue)
    return kept


def _dialogue_text(dialogue: NormalizedDialogue) -> str:
    return " ".join(
        line.content for line in dialogue.lines if line.content
    ).strip()


def _split_dialogue(dialogue: NormalizedDialogue, splitter: SentenceSplitter) -> list[str]:
    text = splitter.join_text(
        [line.content for line in dialogue.lines if line.content]
    ).strip()
    if not text:
        return []
    return [sentence for sentence in splitter.split(text) if sentence]


def _single(
    dialogue: NormalizedDialogue, sentences: list[str], index: int
) -> NormalizedDialogue:
    if not sentences:
        return dialogue.model_copy(update={"index": index})
    return NormalizedDialogue(
        index=index,
        start=dialogue.start,
        end=dialogue.end,
        lines=[DialogueLine(style="Default", content=sentences[0].strip())],
    )


def _allocate(start: float, end: float, sentences: list[str]) -> list[_Cue]:
    duration = end - start
    if len(sentences) == 1:
        return [_Cue(sentences[0], start, end)]
    lengths = [len(sentence) for sentence in sentences]
    total = sum(lengths)
    times = [duration * length / total for length in lengths]
    times = [min(max(t, MIN_DURATION), MAX_DURATION) for t in times]
    _redistribute(times, duration)

    cues: list[_Cue] = []
    cursor = start
    for position, sentence in enumerate(sentences):
        if position == len(sentences) - 1:
            cue_end = end
        else:
            cue_end = round(cursor + times[position], ROUND)
        cues.append(_Cue(sentence, cursor, cue_end))
        cursor = cue_end
    return cues


def _redistribute(times: list[float], duration: float) -> None:
    diff = duration - sum(times)
    for position in range(len(times) - 1, -1, -1):
        if abs(diff) <= _EPSILON:
            return
        if diff > 0:
            room = MAX_DURATION - times[position]
            if room <= _EPSILON:
                continue
            take = min(room, diff)
            times[position] += take
            diff -= take
        else:
            room = times[position] - MIN_DURATION
            if room <= _EPSILON:
                continue
            take = min(room, -diff)
            times[position] -= take
            diff += take


def _merge_short(cues: list[_Cue], splitter: SentenceSplitter) -> list[_Cue]:
    result: list[_Cue] = []
    buffer: list[_Cue] = []
    for cue in cues:
        if cue.end - cue.start >= MERGE_THRESHOLD:
            _flush(buffer, result, splitter)
            result.append(cue)
        elif buffer and cue.end - buffer[0].start <= MERGE_MAX:
            buffer.append(cue)
        else:
            _flush(buffer, result, splitter)
            buffer = [cue]
    _flush(buffer, result, splitter)
    return result


def _flush(
    buffer: list[_Cue], result: list[_Cue], splitter: SentenceSplitter
) -> None:
    if not buffer:
        return
    if len(buffer) == 1:
        result.append(buffer[0])
        return
    text = splitter.join_text([cue.text for cue in buffer])
    result.append(_Cue(text, buffer[0].start, buffer[-1].end))


def _sample_text(normalized: NormalizedSubtitle) -> str:
    parts: list[str] = []
    for dialogue in normalized.dialogue[:50]:
        for line in dialogue.lines:
            parts.append(line.content)
    return " ".join(parts)
