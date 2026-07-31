from __future__ import annotations

import pytest

from src.repair.chunks import build_chunks, belongs_to_body
from src.repair.llm_schema import ProposalError, parse_operations


def test_chunks_have_ten_minute_bodies_and_overlapping_context() -> None:
    chunks = build_chunks(1250, chunk_minutes=10, context_seconds=10)

    assert chunks == [
        chunks[0].__class__(0, 0, 600, 0, 610),
        chunks[0].__class__(1, 600, 1200, 590, 1210),
        chunks[0].__class__(2, 1200, 1250, 1190, 1250),
    ]
    assert belongs_to_body(600, chunks[1])
    assert not belongs_to_body(599.99, chunks[1])
    assert belongs_to_body(1249.99, chunks[2])


def test_llm_operations_are_parsed_with_stable_ids() -> None:
    result = parse_operations(
        {
            "operations": [
                {
                    "subtitle_ids": [7],
                    "asr_ids": [101, 102],
                    "action": "revise",
                    "text": "corrected",
                    "reason": "fixes a negation",
                },
                {
                    "subtitle_ids": [],
                    "asr_ids": [103],
                    "action": "insert",
                    "text": "missing line",
                    "reason": "clear missing speech",
                },
            ]
        },
        subtitle_ids={7},
        asr_ids={101, 102, 103},
        body_subtitle_ids={7},
        body_asr_ids={101, 102, 103},
    )

    assert result[0].subtitle_ids == (7,)
    assert result[0].asr_ids == (101, 102)
    assert result[1].action == "insert"


def test_llm_operations_reject_unknown_or_duplicate_subtitle_ids() -> None:
    with pytest.raises(ProposalError, match="unknown subtitle id"):
        parse_operations(
            {
                "operations": [
                    {
                        "subtitle_ids": [99],
                        "asr_ids": [101],
                        "action": "keep",
                        "text": "",
                        "reason": "",
                    }
                ]
            },
            subtitle_ids={7},
            asr_ids={101},
            body_subtitle_ids={7},
            body_asr_ids={101},
        )

    with pytest.raises(ProposalError, match="duplicate subtitle id"):
        parse_operations(
            {
                "operations": [
                    {
                        "subtitle_ids": [7],
                        "asr_ids": [101],
                        "action": "keep",
                        "text": "",
                        "reason": "",
                    },
                    {
                        "subtitle_ids": [7],
                        "asr_ids": [102],
                        "action": "revise",
                        "text": "changed",
                        "reason": "",
                    },
                ]
            },
            subtitle_ids={7},
            asr_ids={101, 102},
            body_subtitle_ids={7},
            body_asr_ids={101, 102},
        )


def test_llm_cannot_modify_context_only_entries() -> None:
    with pytest.raises(ProposalError, match="outside chunk body"):
        parse_operations(
            {
                "operations": [
                    {
                        "subtitle_ids": [8],
                        "asr_ids": [104],
                        "action": "keep",
                        "text": "",
                        "reason": "",
                    }
                ]
            },
            subtitle_ids={7, 8},
            asr_ids={101, 104},
            body_subtitle_ids={7},
            body_asr_ids={101},
        )


def test_llm_schema_rejects_bad_action_types_and_duplicate_ids_in_one_operation() -> None:
    with pytest.raises(ProposalError, match="unsupported action"):
        parse_operations(
            {
                "operations": [
                    {
                        "subtitle_ids": [7],
                        "asr_ids": [101],
                        "action": [],
                        "text": "",
                        "reason": "",
                    }
                ]
            },
            subtitle_ids={7},
            asr_ids={101},
            body_subtitle_ids={7},
            body_asr_ids={101},
        )


def test_llm_schema_requires_evidence_for_automatic_actions() -> None:
    with pytest.raises(ProposalError, match="requires ASR ids"):
        parse_operations(
            {
                "operations": [
                    {
                        "subtitle_ids": [7],
                        "asr_ids": [],
                        "action": "keep",
                        "text": "",
                        "reason": "",
                    }
                ]
            },
            subtitle_ids={7},
            asr_ids={101},
            body_subtitle_ids={7},
            body_asr_ids={101},
        )


def test_llm_schema_rejects_duplicate_insert_evidence() -> None:
    with pytest.raises(ProposalError, match="duplicate insert ASR id"):
        parse_operations(
            {
                "operations": [
                    {
                        "subtitle_ids": [],
                        "asr_ids": [101],
                        "action": "insert",
                        "text": "first",
                        "reason": "",
                    },
                    {
                        "subtitle_ids": [],
                        "asr_ids": [101],
                        "action": "insert",
                        "text": "second",
                        "reason": "",
                    },
                ]
            },
            subtitle_ids=set(),
            asr_ids={101},
            body_subtitle_ids=set(),
            body_asr_ids={101},
        )

    with pytest.raises(ProposalError, match="duplicate subtitle id"):
        parse_operations(
            {
                "operations": [
                    {
                        "subtitle_ids": [7, 7],
                        "asr_ids": [101],
                        "action": "keep",
                        "text": "",
                        "reason": "",
                    }
                ]
            },
            subtitle_ids={7},
            asr_ids={101},
            body_subtitle_ids={7},
            body_asr_ids={101},
        )
