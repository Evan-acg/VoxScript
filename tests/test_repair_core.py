from __future__ import annotations

from pathlib import Path

import pysubs2

from src.repair.ass_io import (
    load_ass,
    mask_ass_tags,
    restore_ass_tags,
    save_ass,
)
from src.repair.models import AsrSegment, ReviewOperation
from src.repair.operations import apply_operations


ASS_CONTENT = """[Script Info]
Title: Test
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,20,30,1
Style: Alt,Arial,18,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,1,0,0,100,100,0,0,1,2,2,8,11,22,33,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 3,0:00:01.00,0:00:02.00,Alt,Actor,11,22,33,fx,{\\an8}Hello\\N{\\i1}world
Dialogue: 4,0:00:02.50,0:00:04.00,Default,Other,1,2,3,,Next
"""


def write_ass(path: Path) -> None:
    path.write_text(ASS_CONTENT, encoding="utf-8")


def test_ass_round_trip_preserves_event_fields_and_tags(tmp_path: Path) -> None:
    source = tmp_path / "original.ass"
    output = tmp_path / "repaired.ass"
    write_ass(source)

    document = load_ass(source)
    operations = [
        ReviewOperation(
            action="revise",
            subtitle_ids=(1,),
            asr_ids=(101,),
            text="<ASS_TAG_0>Revised<ASS_TAG_1><ASS_TAG_2>world",
            reason="translation correction",
        )
    ]
    asr = [AsrSegment(id=101, start=1.20, end=1.70, text="Hello world")]

    report = apply_operations(document, operations, asr)
    save_ass(document, output)

    assert report.applied == [1]
    assert report.unresolved == []

    saved = pysubs2.load(str(output), encoding="utf-8")
    event = saved.events[0]
    assert event.start == 1100
    assert event.end == 1900
    assert event.text == r"{\an8}Revised\N{\i1}world"
    assert event.layer == 3
    assert event.style == "Alt"
    assert event.name == "Actor"
    assert event.marginl == 11
    assert event.marginr == 22
    assert event.marginv == 33
    assert event.effect == "fx"


def test_ass_loader_reads_utf16_without_bom(tmp_path: Path) -> None:
    source = tmp_path / "original.ass"
    output = tmp_path / "repaired.ass"
    source.write_bytes(ASS_CONTENT.encode("utf-16-le"))

    document = load_ass(source)
    save_ass(document, output)

    assert len(document.events_by_id) == 2
    assert document.events_by_id[1].text.startswith(r"{\an8}")
    assert output.read_bytes().startswith(b"[\x00")


def test_ass_loader_preserves_utf16_big_endian_output(tmp_path: Path) -> None:
    source = tmp_path / "original.ass"
    output = tmp_path / "repaired.ass"
    source.write_bytes(b"\xfe\xff" + ASS_CONTENT.encode("utf-16-be"))

    document = load_ass(source)
    save_ass(document, output)

    assert output.read_bytes().startswith(b"\x00[\x00S")


def test_ass_tag_masking_requires_all_original_tokens() -> None:
    original = r"{\an8}Hello\N{\i1}world"

    masked, tokens = mask_ass_tags(original)

    assert masked == "<ASS_TAG_0>Hello<ASS_TAG_1><ASS_TAG_2>world"
    assert restore_ass_tags(masked, tokens) == original


def test_ass_tag_restoration_rejects_reordered_or_new_tags() -> None:
    _, tokens = mask_ass_tags(r"{\an8}Hello\N")

    import pytest

    with pytest.raises(ValueError):
        restore_ass_tags("<ASS_TAG_1>Hello<ASS_TAG_0>", tokens)

    with pytest.raises(ValueError):
        restore_ass_tags("<ASS_TAG_0>Hello<ASS_TAG_1>{\\i1}", tokens)

    with pytest.raises(ValueError):
        restore_ass_tags("<ASS_TAG_0>Hello<ASS_TAG_1>", ())

    with pytest.raises(ValueError):
        restore_ass_tags("<ASS_TAG_unknown>Hello", ())


def test_delete_and_review_are_kept_without_human_intervention(
    tmp_path: Path,
) -> None:
    source = tmp_path / "original.ass"
    write_ass(source)
    document = load_ass(source)
    original = document.events_by_id[1].text

    report = apply_operations(
        document,
        [
            ReviewOperation(
                action="delete",
                subtitle_ids=(1,),
                asr_ids=(),
                text="",
                reason="no matching speech",
            ),
            ReviewOperation(
                action="review",
                subtitle_ids=(2,),
                asr_ids=(),
                text="",
                reason="ambiguous alignment",
            ),
        ],
        [],
    )

    assert report.applied == []
    assert [item.subtitle_ids for item in report.unresolved] == [(1,), (2,)]
    assert document.events_by_id[1].text == original
    assert len(document.events) == 2


def test_revise_without_original_tags_rejects_generated_ass_tags(tmp_path: Path) -> None:
    source = tmp_path / "original.ass"
    write_ass(source)
    document = load_ass(source)

    report = apply_operations(
        document,
        [
            ReviewOperation(
                action="revise",
                subtitle_ids=(2,),
                asr_ids=(101,),
                text=r"{\i1}generated tag",
                reason="invalid generated tag",
            )
        ],
        [AsrSegment(id=101, start=2.60, end=2.90, text="line")],
    )

    assert report.applied == []
    assert report.unresolved[0].subtitle_ids == (2,)
    assert document.events_by_id[2].text == "Next"


def test_insert_inherits_nearest_dialogue_style(tmp_path: Path) -> None:
    source = tmp_path / "original.ass"
    write_ass(source)
    document = load_ass(source)

    report = apply_operations(
        document,
        [
            ReviewOperation(
                action="insert",
                subtitle_ids=(),
                asr_ids=(102,),
                text="Inserted line",
                reason="missing subtitle",
            )
        ],
        [AsrSegment(id=102, start=4.10, end=4.50, text="Inserted line")],
    )

    assert report.applied == ["insert:102"]
    inserted = document.events[-1]
    assert inserted.text == "Inserted line"
    assert inserted.style == "Default"
    assert inserted.layer == 4
    assert inserted.name == "Other"
    assert inserted.marginl == 1
    assert inserted.marginr == 2
    assert inserted.marginv == 3


def test_timing_is_clamped_before_next_subtitle(tmp_path: Path) -> None:
    source = tmp_path / "original.ass"
    write_ass(source)
    document = load_ass(source)

    report = apply_operations(
        document,
        [
            ReviewOperation(
                action="keep",
                subtitle_ids=(1,),
                asr_ids=(101,),
                text="",
                reason="",
            )
        ],
        [AsrSegment(id=101, start=1.20, end=3.00, text="overlap")],
    )

    assert report.applied == [1]
    assert document.events_by_id[1].start == 1100
    assert document.events_by_id[1].end == 2450
    assert document.events_by_id[1].end < document.events_by_id[2].start


def test_insert_timing_is_also_clamped_before_next_subtitle(tmp_path: Path) -> None:
    source = tmp_path / "original.ass"
    write_ass(source)
    document = load_ass(source)

    report = apply_operations(
        document,
        [
            ReviewOperation(
                action="insert",
                subtitle_ids=(),
                asr_ids=(101,),
                text="Inserted line",
                reason="missing subtitle",
            )
        ],
        [AsrSegment(id=101, start=1.20, end=3.00, text="overlap")],
    )

    assert report.applied == ["insert:101"]
    assert document.events[-1].end == 2450


def test_same_time_dialogues_do_not_clamp_each_other(tmp_path: Path) -> None:
    source = tmp_path / "original.ass"
    source.write_text(
        ASS_CONTENT.replace(
            "Dialogue: 4,0:00:02.50,0:00:04.00,Default,Other,1,2,3,,Next",
            "Dialogue: 4,0:00:01.00,0:00:02.00,Default,Other,1,2,3,,Next",
        ),
        encoding="utf-8",
    )
    document = load_ass(source)

    report = apply_operations(
        document,
        [
            ReviewOperation(
                action="keep",
                subtitle_ids=(1,),
                asr_ids=(101,),
                text="",
                reason="",
            )
        ],
        [AsrSegment(id=101, start=1.20, end=1.70, text="same time")],
    )

    assert report.applied == [1]
    assert document.events_by_id[1].start == 1100
    assert document.events_by_id[1].end == 1900


def test_timing_does_not_extend_past_media_duration(tmp_path: Path) -> None:
    source = tmp_path / "original.ass"
    write_ass(source)
    document = load_ass(source)

    report = apply_operations(
        document,
        [
            ReviewOperation(
                action="keep",
                subtitle_ids=(1,),
                asr_ids=(101,),
                text="",
                reason="",
            )
        ],
        [AsrSegment(id=101, start=1.20, end=2.00, text="bounded")],
        max_end=1.50,
    )

    assert report.applied == [1]
    assert document.events_by_id[1].end == 1500


def test_timing_can_follow_asr_when_original_next_event_is_earlier(
    tmp_path: Path,
) -> None:
    source = tmp_path / "original.ass"
    write_ass(source)
    document = load_ass(source)

    report = apply_operations(
        document,
        [
            ReviewOperation(
                action="keep",
                subtitle_ids=(1,),
                asr_ids=(101,),
                text="",
                reason="",
            )
        ],
        [AsrSegment(id=101, start=2.60, end=2.90, text="late match")],
        clamp_to_next=False,
    )

    assert report.applied == [1]
    assert document.events_by_id[1].start == 2500
    assert document.events_by_id[1].end == 3100


def test_inserted_events_are_tracked_for_later_context(tmp_path: Path) -> None:
    source = tmp_path / "original.ass"
    write_ass(source)
    document = load_ass(source)

    report = apply_operations(
        document,
        [
            ReviewOperation(
                action="insert",
                subtitle_ids=(),
                asr_ids=(101,),
                text="Inserted line",
                reason="missing subtitle",
            )
        ],
        [AsrSegment(id=101, start=601.0, end=601.5, text="Inserted line")],
    )

    assert report.applied == ["insert:101"]
    generated_ids = set(document.events_by_id) - {1, 2}
    assert len(generated_ids) == 1
    generated_id = next(iter(generated_ids))
    assert generated_id < 0
