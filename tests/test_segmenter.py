from __future__ import annotations

from src.repair.models import AsrSegment
from src.repair.segmenter import split_asr_segment


def test_segmenter_splits_sentences_and_preserves_total_time() -> None:
    segment = AsrSegment(
        id=1,
        start=10.0,
        end=14.0,
        text="First sentence. Second sentence!",
    )

    parts = split_asr_segment(segment)

    assert [part.text for part in parts] == ["First sentence.", "Second sentence!"]
    assert parts[0].start == 10.0
    assert parts[-1].end == 14.0
    assert parts[0].end == parts[1].start


def test_segmenter_keeps_text_without_sentence_punctuation() -> None:
    segment = AsrSegment(id=1, start=2.0, end=3.0, text="unfinished phrase")

    assert split_asr_segment(segment) == [segment]
