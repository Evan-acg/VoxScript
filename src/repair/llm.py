from __future__ import annotations

import json
import os
from typing import Any

from ..config import get, get_float, get_int
from ..progress import ProgressCallback, ProgressEvent, null_callback
from .llm_schema import ProposalError, parse_operations
from .models import ReviewOperation


class LLMError(Exception):
    pass


DEFAULT_SYSTEM_PROMPT = """你是字幕自动修复助手。
你需要把带时间的 ASR 语音片段和原始字幕进行匹配，并检查字幕文本。
ASR 只提供源语言语音证据，不能直接替换原字幕语言。
没有明确错误时使用 keep，不要为了文风统一重写文本。
重点检查漏译、误译、否定、数字、人名、术语和指代。
只修改当前分段主体区域，context 条目只能用于理解。
不要修改 ASS 标签占位符，不要生成时间戳。
输出 JSON 对象，格式为 {"operations": [{"subtitle_ids": [], "asr_ids": [], "action": "keep|revise|insert|delete|review", "text": "", "reason": ""}]}。
delete 和 review 只用于表达不确定判断，程序会保守保留原字幕。
"""


class LLMClient:
    def __init__(
        self,
        model: str,
        base_url: str = "",
        *,
        api_key: str | None = None,
        client: Any | None = None,
        max_retries: int = 2,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            key = api_key or os.environ.get("VOX_API_KEY") or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise LLMError("VOX_API_KEY or OPENAI_API_KEY environment variable is not set")
            from openai import OpenAI

            kwargs: dict[str, Any] = {"api_key": key}
            if base_url:
                kwargs["base_url"] = base_url
            self._client = OpenAI(**kwargs)
        self._model = model
        self._max_retries = max(0, max_retries)

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
        on_progress: ProgressCallback = null_callback,
    ) -> list[ReviewOperation]:
        on_progress(ProgressEvent("llm", 0, 1, f"Analysing chunk {chunk_id}"))
        messages = [
            {"role": "system", "content": get("prompts", "system_prompt", fallback=DEFAULT_SYSTEM_PROMPT)},
            {"role": "user", "content": self._build_user_message(
                chunk_id=chunk_id,
                subtitle_entries=subtitle_entries,
                asr_entries=asr_entries,
                source_language=source_language,
                target_language=target_language,
            )},
        ]
        last_error: LLMError | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=get_float("llm", "temperature", fallback=0.0),
                    max_tokens=get_int("llm", "max_tokens", fallback=32768),
                    response_format={"type": "json_object"},
                    stream=False,
                )
                content = self._response_content(response)
                payload = self._parse_response(content)
                try:
                    operations = parse_operations(
                        payload,
                        subtitle_ids={entry["id"] for entry in subtitle_entries},
                        asr_ids={entry["id"] for entry in asr_entries},
                        body_subtitle_ids=body_subtitle_ids,
                        body_asr_ids=body_asr_ids,
                    )
                except ProposalError as error:
                    raise LLMError(str(error)) from error
                on_progress(ProgressEvent("llm", 1, 1, f"Chunk {chunk_id} analysed"))
                return operations
            except LLMError as error:
                last_error = error
            except Exception as error:
                last_error = LLMError(f"LLM request failed: {error}")
            if attempt < self._max_retries:
                on_progress(
                    ProgressEvent(
                        "llm",
                        0,
                        1,
                        f"Retrying chunk {chunk_id} ({attempt + 1}/{self._max_retries})",
                    )
                )
        raise last_error or LLMError("LLM request failed")

    @staticmethod
    def _build_user_message(
        *,
        chunk_id: int,
        subtitle_entries: list[dict[str, Any]],
        asr_entries: list[dict[str, Any]],
        source_language: str | None,
        target_language: str,
    ) -> str:
        values = {
            "chunk_id": chunk_id,
            "source_language": source_language or "auto",
            "target_language": target_language,
            "subtitle_entries": json.dumps(subtitle_entries, ensure_ascii=False, indent=2),
            "asr_entries": json.dumps(asr_entries, ensure_ascii=False, indent=2),
        }
        template = get("prompts", "user_prompt")
        if template:
            try:
                return template.format(**values)
            except (KeyError, ValueError):
                pass
        return "\n".join(
            [
                f"chunk_id={chunk_id}",
                f"source_language={source_language or 'auto'}",
                f"target_language={target_language}",
                "字幕条目 JSON:",
                values["subtitle_entries"],
                "ASR 条目 JSON:",
                values["asr_entries"],
                "只输出操作 JSON，不要输出 Markdown 或其他说明。",
            ]
        )

    @staticmethod
    def _response_content(response: Any) -> str:
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, KeyError, TypeError) as error:
            raise LLMError(f"LLM response has no message content: {error}") from error
        if not isinstance(content, str) or not content.strip():
            raise LLMError("LLM returned empty content")
        return content.strip()

    @staticmethod
    def _parse_response(content: str) -> dict[str, Any]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as error:
            raise LLMError(f"failed to parse LLM response: {error}") from error
        if not isinstance(payload, dict):
            raise LLMError("LLM response must be a JSON object")
        return payload
