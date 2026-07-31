from __future__ import annotations

import json
from pathlib import Path

import pysubs2
import pytest

from src.repair.models import AsrSegment, ReviewOperation, SubtitleMatch
from src.repair.workflow import (
    RepairError,
    RepairOptions,
    _expand_operation_evidence,
    run_repair,
)


ASS_CONTENT = """[Script Info]
Title: Workflow Test
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,20,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 2,0:00:01.00,0:00:02.00,Default,Actor,1,2,3,,Original text
"""


class FakeAudio:
    def get_duration(self, input_path: str) -> float:
        return 5.0

    def extract(self, input_path: str, output_path: str, **kwargs: object) -> None:
        Path(output_path).write_bytes(b"audio")


class LongFakeAudio(FakeAudio):
    def get_duration(self, input_path: str) -> float:
        return 605.0


class FakeAsr:
    def transcribe(self, audio_path: str, *, offset: float, chunk_id: int, **kwargs: object) -> list[AsrSegment]:
        return [
            AsrSegment(
                id=0,
                start=offset + 1.20,
                end=offset + 1.70,
                text="Correct source text",
                chunk_id=chunk_id,
            )
        ]


class CountingAsr(FakeAsr):
    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, *args: object, **kwargs: object) -> list[AsrSegment]:
        self.calls += 1
        return super().transcribe(*args, **kwargs)


class FakeLlm:
    def __init__(self, action: str) -> None:
        self.action = action

    def propose(self, **kwargs: object) -> list[ReviewOperation]:
        subtitle_id = next(iter(kwargs["body_subtitle_ids"]))
        asr_id = next(iter(kwargs["body_asr_ids"]))
        return [
            ReviewOperation(
                action=self.action,
                subtitle_ids=(subtitle_id,),
                asr_ids=(asr_id,),
                text="Automatically corrected",
                reason="test proposal",
            )
        ]


class FailingLlm:
    def propose(self, **kwargs: object) -> list[ReviewOperation]:
        raise RuntimeError("temporary LLM failure")


def test_workflow_writes_repaired_ass_without_touching_original(tmp_path: Path) -> None:
    video = tmp_path / "video.mkv"
    original = tmp_path / "original.ass"
    output = tmp_path / "repaired.ass"
    video.write_bytes(b"video")
    original.write_text(ASS_CONTENT, encoding="utf-8")

    result = run_repair(
        video,
        original,
        output,
        audio=FakeAudio(),
        asr=FakeAsr(),
        llm=FakeLlm("revise"),
        options=RepairOptions(work_dir=tmp_path / "work", keep_artifacts=True),
    )

    assert result.output_path == output
    assert output.exists()
    assert original.read_text(encoding="utf-8") == ASS_CONTENT

    repaired = pysubs2.load(str(output), encoding="utf-8")
    assert repaired.events[0].text == "Automatically corrected"
    assert repaired.events[0].start == 1100
    assert repaired.events[0].end == 1900

    asr_payload = json.loads((tmp_path / "work" / "asr.json").read_text(encoding="utf-8"))
    report_payload = json.loads((tmp_path / "work" / "review.json").read_text(encoding="utf-8"))
    assert len(asr_payload["segments"]) == 1
    assert report_payload["applied"] == [1]
    assert report_payload["unresolved"] == []


def test_workflow_keeps_delete_proposals_and_records_them(tmp_path: Path) -> None:
    video = tmp_path / "video.mkv"
    original = tmp_path / "original.ass"
    output = tmp_path / "repaired.ass"
    video.write_bytes(b"video")
    original.write_text(ASS_CONTENT, encoding="utf-8")

    result = run_repair(
        video,
        original,
        output,
        audio=FakeAudio(),
        asr=FakeAsr(),
        llm=FakeLlm("delete"),
        options=RepairOptions(work_dir=tmp_path / "work", keep_artifacts=True),
    )

    assert result.partial
    repaired = pysubs2.load(str(output), encoding="utf-8")
    assert len(repaired.events) == 1
    assert repaired.events[0].text == "Original text"

    report_payload = json.loads((tmp_path / "work" / "review.json").read_text(encoding="utf-8"))
    assert report_payload["unresolved"][0]["action"] == "delete"


def test_workflow_keeps_report_when_automatic_fallback_was_used(tmp_path: Path) -> None:
    video = tmp_path / "video.mkv"
    original = tmp_path / "original.ass"
    output = tmp_path / "repaired.ass"
    video.write_bytes(b"video")
    original.write_text(ASS_CONTENT, encoding="utf-8")

    result = run_repair(
        video,
        original,
        output,
        audio=FakeAudio(),
        asr=FakeAsr(),
        llm=FakeLlm("delete"),
    )

    assert result.review_path is not None
    assert result.review_path.exists()


def test_workflow_keeps_asr_result_when_llm_fails(tmp_path: Path) -> None:
    video = tmp_path / "video.mkv"
    original = tmp_path / "original.ass"
    output = tmp_path / "repaired.ass"
    video.write_bytes(b"video")
    original.write_text(ASS_CONTENT, encoding="utf-8")

    result = run_repair(
        video,
        original,
        output,
        audio=FakeAudio(),
        asr=FakeAsr(),
        llm=FailingLlm(),
        options=RepairOptions(work_dir=tmp_path / "work", keep_artifacts=True),
    )

    assert result.partial
    assert output.exists()
    report_payload = json.loads((tmp_path / "work" / "review.json").read_text(encoding="utf-8"))
    assert report_payload["errors"][0]["message"] == "temporary LLM failure"


def test_workflow_rejects_overwriting_original_ass(tmp_path: Path) -> None:
    video = tmp_path / "video.mkv"
    original = tmp_path / "original.ass"
    video.write_bytes(b"video")
    original.write_text(ASS_CONTENT, encoding="utf-8")

    with pytest.raises(RepairError, match="output must not overwrite"):
        run_repair(
            video,
            original,
            original,
            audio=FakeAudio(),
            asr=FakeAsr(),
            llm=FakeLlm("keep"),
        )


def test_workflow_runs_asr_once_for_multiple_llm_chunks(tmp_path: Path) -> None:
    video = tmp_path / "video.mkv"
    original = tmp_path / "original.ass"
    output = tmp_path / "repaired.ass"
    video.write_bytes(b"video")
    original.write_text(ASS_CONTENT, encoding="utf-8")
    asr = CountingAsr()

    run_repair(
        video,
        original,
        output,
        audio=LongFakeAudio(),
        asr=asr,
        llm=FakeLlm("keep"),
        options=RepairOptions(work_dir=tmp_path / "work", keep_artifacts=True),
    )

    assert asr.calls == 1


def test_revision_keeps_full_deterministic_asr_span() -> None:
    operation = ReviewOperation(
        action="revise",
        subtitle_ids=(1,),
        asr_ids=(2,),
        text="corrected",
        reason="text correction",
    )

    expanded = _expand_operation_evidence(
        [operation],
        {1: SubtitleMatch(1, (1, 2), 0.95)},
    )

    assert expanded[0].asr_ids == (1, 2)
