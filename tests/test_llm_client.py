from __future__ import annotations

import json

import pytest

from src.repair.llm import LLMClient, LLMError


class FakeCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        return type(
            "Response",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {"message": type("Message", (), {"content": self.content})()},
                    )()
                ]
            },
        )()


class FakeClient:
    def __init__(self, content: str) -> None:
        self.completions = FakeCompletions(content)
        self.chat = type("Chat", (), {"completions": self.completions})()


class RetryingCompletions(FakeCompletions):
    def __init__(self, contents: list[str]) -> None:
        super().__init__(contents[0])
        self.contents = contents
        self.calls = 0

    def create(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        content = self.contents[min(self.calls, len(self.contents) - 1)]
        self.calls += 1
        return type(
            "Response",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {"message": type("Message", (), {"content": content})()},
                    )()
                ]
            },
        )()


class RetryingClient:
    def __init__(self, completions: RetryingCompletions) -> None:
        self.completions = completions
        self.chat = type("Chat", (), {"completions": completions})()


def test_llm_client_returns_validated_operations_and_sends_chunk_context() -> None:
    payload = {
        "operations": [
            {
                "subtitle_ids": [7],
                "asr_ids": [101],
                "action": "keep",
                "text": "",
                "reason": "aligned",
            }
        ]
    }
    client = FakeClient(json.dumps(payload))
    llm = LLMClient(model="test-model", client=client)

    operations = llm.propose(
        chunk_id=2,
        subtitle_entries=[{"id": 7, "text": "<ASS_TAG_0>Hello", "scope": "body"}],
        asr_entries=[{"id": 101, "text": "Hello", "scope": "body"}],
        body_subtitle_ids={7},
        body_asr_ids={101},
        source_language="en",
        target_language="zh",
    )

    assert operations[0].action == "keep"
    request = client.completions.kwargs
    assert request["response_format"] == {"type": "json_object"}
    user_message = request["messages"][1]["content"]
    assert "<ASS_TAG_0>Hello" in user_message
    assert "chunk_id=2" in user_message


def test_llm_client_rejects_empty_or_invalid_response() -> None:
    llm = LLMClient(model="test-model", client=FakeClient("not json"))

    with pytest.raises(LLMError, match="failed to parse"):
        llm.propose(
            chunk_id=0,
            subtitle_entries=[],
            asr_entries=[],
            body_subtitle_ids=set(),
            body_asr_ids=set(),
            source_language=None,
            target_language="auto",
        )


def test_llm_client_retries_empty_response() -> None:
    payload = json.dumps(
        {
            "operations": [
                {
                    "subtitle_ids": [7],
                    "asr_ids": [101],
                    "action": "keep",
                    "text": "",
                    "reason": "aligned",
                }
            ]
        }
    )
    completions = RetryingCompletions(["", payload])
    llm = LLMClient(
        model="test-model",
        client=RetryingClient(completions),
        max_retries=1,
    )

    operations = llm.propose(
        chunk_id=0,
        subtitle_entries=[{"id": 7, "text": "Hello", "scope": "body"}],
        asr_entries=[{"id": 101, "text": "Hello", "scope": "body"}],
        body_subtitle_ids={7},
        body_asr_ids={101},
        source_language="en",
        target_language="zh",
    )

    assert operations[0].action == "keep"
    assert completions.calls == 2
