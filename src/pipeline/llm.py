from __future__ import annotations

import json
import os

from openai import OpenAI

from ..config import get, get_float, get_int
from ..progress import ProgressCallback, ProgressEvent, null_callback


class ProofreadError(Exception):
    pass


def _build_system_prompt() -> str:
    return get("prompts", "system_prompt")


def _build_user_prompt(
    original_subtitle: str, ref_transcript: str, subtitle_format: str
) -> str:
    template = get("prompts", "user_prompt")
    if not template:
        template = "\u7528\u6237\u539f\u59cb\u5b57\u5e55\uff08{format} \u683c\u5f0f\uff09\n\n{subtitle}\n\n## WhisperX \u53c2\u8003\u8f6c\u5f55\uff08SRT \u683c\u5f0f\uff09\n\n{transcript}"
    return template.format(
        format=subtitle_format.upper(),
        subtitle=original_subtitle,
        transcript=ref_transcript,
    )


def _parse_json_response(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        for line in cleaned.splitlines():
            if line.startswith("```"):
                cleaned = cleaned.removeprefix(line).strip()
                break
        if cleaned.endswith("```"):
            cleaned = cleaned.removesuffix("```").strip()

    cleaned = cleaned.removeprefix("json").strip()
    return json.loads(cleaned)


class LLMClient:
    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str = "",
    ) -> None:
        api_key = os.environ.get("VOX_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ProofreadError(
                "VOX_API_KEY or OPENAI_API_KEY environment variable is not set"
            )
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._model = model
        from loguru import logger
        logger.info(f"LLM: model={model}, base_url={base_url or 'default'}")

    def proofread(
        self,
        original_subtitle: str,
        ref_transcript: str,
        subtitle_format: str,
        on_progress: ProgressCallback = null_callback,
    ) -> dict:
        on_progress(
            ProgressEvent("llm", 0, 1, "Sending to LLM for proofreading...")
        )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _build_system_prompt()},
                {
                    "role": "user",
                    "content": _build_user_prompt(
                        original_subtitle, ref_transcript, subtitle_format
                    ),
                },
            ],
            temperature=get_float("llm", "temperature", fallback=0.0),
            max_tokens=get_int("llm", "max_tokens", fallback=4096),
        )

        on_progress(
            ProgressEvent("llm", 1, 1, "LLM proofreading complete")
        )

        content = response.choices[0].message.content
        if not content:
            reason = response.choices[0].finish_reason
            usage_dict = dict(response.usage or {})
            raise ProofreadError(
                f"LLM returned empty content\n"
                f"  model={response.model}\n"
                f"  finish_reason={reason}\n"
                f"  usage={usage_dict}"
            )

        try:
            return _parse_json_response(content)
        except (json.JSONDecodeError, KeyError) as e:
            preview_chars = get_int(
                "llm", "response_preview_chars", fallback=500
            )
            raise ProofreadError(
                f"Failed to parse LLM response: {e}\nRaw: {content[:preview_chars]}"
            )
