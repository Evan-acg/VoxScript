from __future__ import annotations

from src.repair.matcher import match_cues_to_asr
from src.repair.models import AsrSegment, SubtitleCue


def test_matcher_uses_english_line_from_bilingual_ass() -> None:
    cues = [
        SubtitleCue(
            id=1,
            event_index=0,
            start=100.0,
            end=102.0,
            text=r"中文台词\N{\rEng}I could have kissed her.",
        )
    ]
    asr = [
        AsrSegment(id=7, start=108.0, end=109.0, text="I could have kissed her.")
    ]

    matches = match_cues_to_asr(cues, asr)

    assert matches[1].asr_ids == (7,)
    assert matches[1].score > 0.9


def test_matcher_handles_one_subtitle_spanning_multiple_asr_segments() -> None:
    cues = [
        SubtitleCue(
            id=1,
            event_index=0,
            start=10.0,
            end=12.0,
            text=r"中文\N{\rEng}This is a longer sentence.",
        )
    ]
    asr = [
        AsrSegment(id=1, start=10.5, end=10.9, text="This is a longer"),
        AsrSegment(id=2, start=11.0, end=11.6, text="sentence."),
    ]

    matches = match_cues_to_asr(cues, asr)

    assert matches[1].asr_ids == (1, 2)


def test_matcher_allows_two_split_subtitles_to_share_one_asr_segment() -> None:
    cues = [
        SubtitleCue(1, 0, 10.0, 11.0, r"中文\N{\rEng}This is"),
        SubtitleCue(2, 1, 11.0, 12.0, r"中文\N{\rEng}a sentence."),
    ]
    asr = [AsrSegment(id=1, start=10.2, end=11.8, text="This is a sentence.")]

    matches = match_cues_to_asr(cues, asr)

    assert matches[1].asr_ids == (1,)
    assert matches[2].asr_ids == (1,)


def test_matcher_does_not_match_screen_text_without_source_speech() -> None:
    cues = [
        SubtitleCue(
            id=1,
            event_index=0,
            start=20.0,
            end=22.0,
            text=r"{\an8}HIMYM字幕组 典藏版",
        )
    ]
    asr = [AsrSegment(id=1, start=20.0, end=21.0, text="unrelated dialogue")]

    matches = match_cues_to_asr(cues, asr)

    assert matches == {}
