from __future__ import annotations

import math
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.entity.subtitle import DialogueLine, NormalizedSubtitle
from src.entity.translate import LLMConfig
from src.llm.base import LLMProvider

_CJK_WRAP = 22
_LATIN_WRAP = 42
_SCENE_GAP = 1.5
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0
_RETRY_TIMEOUT = 180.0
_MID_CONTEXT_SIZE = 3
_SAMPLE_MIN_RATIO = 0.05
_TOKEN_BUDGET_PER_CUE = 64
_TOKEN_BUDGET_FACTOR = 1.6
_TOKEN_BUDGET_MIN = 512
_TOKEN_BUDGET_MAX = 4096
_CJK_PUNCT = "、，。！？；"
_CJK_LANGS = ("zh", "ja", "ko")
_SYMBOL_RE = re.compile(r"^[\s.,!?;:·、，。！？…~\-—()\[\]{}「」『』""'']*$")
_DIGIT_RE = re.compile(r"^[\d\s:.,/\\\-]+$")
_SFX_RE = re.compile(r"^[([{【［][^()[\]{}【】]*[)\]】｝][\s.!。]*$")
_CREDIT_RE = re.compile(
    r"翻译|字幕|压制|校对|校译|片源|staff|translated|subtitles|encoding|typeset",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TranslateResult:
    translated: NormalizedSubtitle
    report: str


@dataclass
class _Cue:
    position: int
    dialogue_index: int
    start: float
    end: float
    source: str
    source_parts: int
    old_target: str
    skip: bool = False
    action: str = "kept"
    parts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _BatchOutcome:
    index: int
    status: str
    elapsed: float
    prompt_tokens: int
    completion_tokens: int
    failed: bool = False


class _TokenCount:
    def __init__(self) -> None:
        self.prompt = 0
        self.completion = 0
        self.requests = 0

    def add(self, prompt: int, completion: int) -> None:
        self.prompt += prompt
        self.completion += completion
        self.requests += 1


def translate_subtitle_file(
    input_path: Path,
    output_path: Path | None = None,
    report_path: Path | None = None,
    config: LLMConfig | None = None,
    provider: LLMProvider | None = None,
    source_style: str = "Eng",
    on_log: Callable[[str], None] | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> TranslateResult:
    if config is None or provider is None:
        raise RuntimeError("LLM config and provider are required for translation")
    normalized = NormalizedSubtitle.model_validate_json(
        input_path.read_text(encoding="utf-8")
    )
    result = translate_subtitle(
        normalized,
        config,
        provider,
        source_style=source_style,
        on_log=on_log,
        on_progress=on_progress,
    )
    if output_path is not None:
        text = result.translated.model_dump_json(indent=2, ensure_ascii=False) + "\n"
        output_path.write_text(text, encoding="utf-8")
    if report_path is not None:
        report_path.write_text(result.report + "\n", encoding="utf-8")
    return result


def translate_subtitle(
    normalized: NormalizedSubtitle,
    config: LLMConfig,
    provider: LLMProvider,
    source_style: str = "Eng",
    on_log: Callable[[str], None] | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> TranslateResult:
    cjk = _is_cjk(config.target_lang)
    report = _Report()
    report.header(normalized.path)

    cues = _extract_cues(normalized, source_style, config.target_style)
    pending = [cue for cue in cues if not cue.skip]
    report.inputs(len(cues), len(pending))
    if not pending:
        report.summary(cues, tokens=_TokenCount())
        return TranslateResult(
            translated=_build_output(normalized, cues, config.target_style),
            report=report.render(),
        )

    batches = _chunk(
        pending, config.max_lines_per_request, scene_gaps=config.sequential
    )
    tokens = _TokenCount()
    if config.sequential:
        _run_sequential(
            batches, config, provider, cjk, tokens, report, on_log,
            on_progress, len(pending),
        )
    else:
        _run_parallel(
            batches, config, provider, cjk, tokens, report, on_log,
            on_progress, len(pending),
        )

    _unify_aliases(cues, config.alias_groups, config.term_base, report)
    report.summary(cues, tokens)
    report.tokens(tokens)
    _sample(cues, report)
    return TranslateResult(
        translated=_build_output(normalized, cues, config.target_style),
        report=report.render(),
    )


def _run_sequential(
    batches: list[list[_Cue]],
    config: LLMConfig,
    provider: LLMProvider,
    cjk: bool,
    tokens: _TokenCount,
    report: _Report,
    on_log: Callable[[str], None] | None,
    on_progress: Callable[[float], None] | None,
    pending_count: int,
) -> None:
    prev_tail: list[tuple[str, str]] = []
    done = 0
    total_elapsed = 0.0
    for index, batch in enumerate(batches, start=1):
        outcome = _translate_batch(
            batch, index, config, provider, cjk, prev_tail, False, report
        )
        prev_tail = _cue_tail(batch)
        _finalize_batch(outcome, batch, tokens, report, on_log)
        total_elapsed += outcome.elapsed
        done += len(batch)
        if on_progress is not None:
            on_progress(done / pending_count * 100)
    report.api_time(total_elapsed)


def _run_parallel(
    batches: list[list[_Cue]],
    config: LLMConfig,
    provider: LLMProvider,
    cjk: bool,
    tokens: _TokenCount,
    report: _Report,
    on_log: Callable[[str], None] | None,
    on_progress: Callable[[float], None] | None,
    pending_count: int,
) -> None:
    outcomes: list[_BatchOutcome] = []
    done = 0
    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        future_to_index = {
            pool.submit(
                _translate_batch,
                batch,
                index,
                config,
                provider,
                cjk,
                _source_tail(batches, index - 1),
                True,
                report,
            ): index
            for index, batch in enumerate(batches, start=1)
        }
        for future in as_completed(future_to_index):
            outcome = future.result()
            outcomes.append(outcome)
            done += len(batches[outcome.index - 1])
            if on_progress is not None:
                on_progress(done / pending_count * 100)
    total_elapsed = 0.0
    for outcome in sorted(outcomes, key=lambda item: item.index):
        _finalize_batch(outcome, batches[outcome.index - 1], tokens, report, on_log)
        total_elapsed += outcome.elapsed
    report.api_time(total_elapsed)


def _finalize_batch(
    outcome: _BatchOutcome,
    batch: list[_Cue],
    tokens: _TokenCount,
    report: _Report,
    on_log: Callable[[str], None] | None,
) -> None:
    tokens.add(outcome.prompt_tokens, outcome.completion_tokens)
    report.batch(
        outcome.index, batch, outcome.status, outcome.elapsed,
        outcome.prompt_tokens, outcome.completion_tokens,
    )
    if outcome.failed:
        _notify(
            on_log,
            f"batch {outcome.index}: all retries failed, keeping old translations",
        )


def _extract_cues(
    normalized: NormalizedSubtitle, source_style: str, target_style: str
) -> list[_Cue]:
    cues: list[_Cue] = []
    for position, dialogue in enumerate(normalized.dialogue):
        source = [
            line.content
            for line in dialogue.lines
            if line.style == source_style and line.content
        ]
        target = [
            line.content
            for line in dialogue.lines
            if line.style == target_style and line.content
        ]
        source_text = "\\N".join(source)
        cues.append(
            _Cue(
                position=position,
                dialogue_index=dialogue.index,
                start=dialogue.start,
                end=dialogue.end,
                source=source_text,
                source_parts=len(source),
                old_target="\\N".join(target),
                skip=_is_skippable(source_text),
            )
        )
    return cues


def _is_skippable(source: str) -> bool:
    text = source.strip()
    if not text:
        return True
    if _SYMBOL_RE.match(text) or _DIGIT_RE.match(text):
        return True
    if _SFX_RE.match(text):
        return True
    if _CREDIT_RE.search(text):
        return True
    return False


def _chunk(
    cues: list[_Cue], max_lines: int, scene_gaps: bool = True
) -> list[list[_Cue]]:
    batches: list[list[_Cue]] = []
    cursor = 0
    while cursor < len(cues):
        end = min(cursor + max_lines, len(cues))
        if scene_gaps:
            boundary = cursor
            for i in range(cursor + 1, end):
                if cues[i].start - cues[i - 1].end > _SCENE_GAP:
                    boundary = i
            if boundary > cursor:
                end = boundary
        batches.append(cues[cursor:end])
        cursor = end
    return batches


def _translate_batch(
    batch: list[_Cue],
    index: int,
    config: LLMConfig,
    provider: LLMProvider,
    cjk: bool,
    prev_tail: list[tuple[str, str]],
    parallel: bool,
    report: _Report,
) -> _BatchOutcome:
    start = time.perf_counter()
    texts = [cue.source for cue in batch]
    prompt = _build_prompt(config, texts, prev_tail, parallel)
    max_tokens = _completion_budget(batch)
    content, error, prompt_tokens, completion_tokens = _call_with_retry(
        provider, config, prompt, max_tokens
    )
    if content is None:
        for cue in batch:
            cue.action = "failed"
        return _BatchOutcome(
            index=index,
            status=f"FAILED ({error})",
            elapsed=time.perf_counter() - start,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            failed=True,
        )

    lines = _split_response(content)
    count = len(batch)
    status = "OK"
    if len(lines) < count:
        missing = count - len(lines)
        report.warning(
            f"batch {index}: response had {len(lines)} lines for {count} cues; "
            "missing positions padded with old translation"
        )
        lines += [""] * missing
        status = f"PADDED (missing {missing})"
    elif len(lines) > count:
        extra = len(lines) - count
        report.warning(
            f"batch {index}: response had {len(lines)} lines for {count} cues; "
            "extra lines truncated"
        )
        lines = lines[:count]
        status = f"TRUNCATED (extra {extra})"
    for cue, line in zip(batch, lines, strict=True):
        _post_process(cue, line, config, cjk, report)
    return _BatchOutcome(
        index=index,
        status=status,
        elapsed=time.perf_counter() - start,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _build_prompt(
    config: LLMConfig,
    texts: list[str],
    prev_tail: list[tuple[str, str]],
    parallel: bool,
) -> list[dict[str, str]]:
    system_parts = [
        f"你是一位专业的影视字幕翻译员，精通 {config.source_lang} 与 {config.target_lang}。",
        f"你的任务是将以下 {config.source_lang} 字幕翻译为 {config.target_lang}。",
        "翻译须遵循以下标准：",
        "1. 信：准确传达原意，无错译、漏译。",
        "2. 达：译文通顺流畅，完全符合目标语言的口语表达习惯。",
        "3. 雅：译文优雅得体，但不过分书面化，保持原句的口语风格与情绪色彩。",
        "",
        "输出规则（必须严格遵守）：",
        "- 行数与输入句数严格一致，一行对应一句，按输入顺序输出。",
        "- 只输出译文本身；禁止序号、编号、标题、注释、解释、空行、原文重复、JSON 或任何额外内容。",
        "- 若某句确实无法翻译，也必须输出一行，内容为该句原文。",
        "",
        "风格要求：",
        '- 译文必须接地气，像真实人物在说话，避免"翻译腔"（如避免生硬的直译，避免滥用"哦"、"啊"、"的"等助词）。',
        "- 贴合原句的语域（正式/非正式、幽默/严肃、愤怒/温柔），并在译文中体现。",
        "- 译文应能直接匹配字幕显示时长，若某句译文过长，请在适当位置用 \\N 自行断行。",
        "- 源文本中的 \\N 表示强制换行，译文应在对应位置保留。",
    ]
    if config.summary or config.characters or config.relationships:
        system_parts.append("")
        system_parts.append("以下是本集/本段对话的背景信息：")
        for section in (config.summary, config.characters, config.relationships):
            if section:
                system_parts.append(section)
        system_parts.append("请在翻译时保持各角色的说话风格一致。")
    if config.term_base:
        system_parts.append("")
        system_parts.append("术语表（必须严格遵守，不得直译）：")
        for source, target in config.term_base.items():
            system_parts.append(f"{source}: {target}")

    user_parts: list[str] = []
    if prev_tail:
        if parallel:
            user_parts.append("以下是前文内容（仅供连贯参考，不需重译）：")
            for source, _ in prev_tail:
                user_parts.append(f"源文: {source}")
        else:
            user_parts.append(
                "以下是前文最后几句（已翻译，仅供参考，不需重译），用于保持连贯："
            )
            for source, target in prev_tail:
                user_parts.append(f"源文: {source}")
                user_parts.append(f"译文: {target}")
        user_parts.append("")
    user_parts.append(f"请将以下 {config.source_lang} 句子逐句翻译为 {config.target_lang}。")
    user_parts.append("每行一个句子，按顺序输出译文，不要添加任何序号、注释或额外说明。")
    user_parts.append("待翻译文本：")
    user_parts.extend(texts)
    return [
        {"role": "system", "content": "\n".join(system_parts)},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def _completion_budget(batch: list[_Cue]) -> int:
    chars = sum(len(cue.source) for cue in batch)
    budget = max(
        int(chars * _TOKEN_BUDGET_FACTOR),
        len(batch) * _TOKEN_BUDGET_PER_CUE,
    )
    return min(max(budget, _TOKEN_BUDGET_MIN), _TOKEN_BUDGET_MAX)


def _call_with_retry(
    provider: LLMProvider,
    config: LLMConfig,
    prompt: list[dict[str, str]],
    max_tokens: int,
) -> tuple[str | None, str | None, int, int]:
    last_error: str | None = None
    prompt_tokens = 0
    completion_tokens = 0
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            result = provider.chat(
                prompt, config.model, config.temperature, _RETRY_TIMEOUT, max_tokens
            )
            prompt_tokens += result.prompt_tokens
            completion_tokens += result.completion_tokens
            content = result.content.strip()
            if not content:
                raise RuntimeError("empty LLM response")
            return content, None, prompt_tokens, completion_tokens
        except RuntimeError as exc:
            last_error = str(exc)
            if attempt < _MAX_RETRIES:
                time.sleep(_BACKOFF_BASE * attempt)
    return None, last_error, prompt_tokens, completion_tokens


def _split_response(content: str) -> list[str]:
    lines = [line.strip() for line in content.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def _post_process(
    cue: _Cue, line: str, config: LLMConfig, cjk: bool, report: _Report
) -> None:
    text = line.strip()
    if not text:
        cue.action = "kept"
        report.warning(
            f"cue #{cue.dialogue_index}: empty translation, old translation kept"
        )
        return
    text = _mirror_breaks(cue, text, cjk, report)
    text = _apply_terms(cue, text, config.term_base, report)
    parts = [part.strip() for part in text.split("\\N") if part.strip()]
    cue.parts = parts or [text]
    cue.action = "translated"


def _mirror_breaks(cue: _Cue, text: str, cjk: bool, report: _Report) -> str:
    if "\\N" in text:
        parts = [part for part in text.split("\\N")]
        if len(parts) == cue.source_parts:
            return text
        report.record(
            f"cue #{cue.dialogue_index}: LLM used {len(parts)} line breaks, "
            f"source has {cue.source_parts}; redistributed"
        )
        text = "".join(parts)
    if cue.source_parts > 1:
        parts = _split_parts(text, cue.source_parts, cjk)
        return "\\N".join(parts)
    wrapped = _wrap_translation(text, cjk)
    if len(wrapped) > 1:
        report.length(
            cue.dialogue_index, len(text), sum(len(p) for p in wrapped),
            f"auto-wrapped into {len(wrapped)} lines",
        )
    return "\\N".join(wrapped)


def _apply_terms(
    cue: _Cue, text: str, term_base: dict[str, str], report: _Report
) -> str:
    result = text
    for source_term, target_term in term_base.items():
        if not source_term or not target_term:
            continue
        if source_term not in cue.source:
            continue
        if target_term in result:
            continue
        if source_term in result:
            result = _replace_term(result, source_term, target_term)
            report.term(cue.dialogue_index, source_term, target_term)
        else:
            report.warning(
                f"cue #{cue.dialogue_index}: source contains '{source_term}' but "
                f"translation lacks the required term '{target_term}'"
            )
    return result


def _replace_term(text: str, source_term: str, target_term: str) -> str:
    if re.match(r"^[A-Za-z]", source_term):
        return re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(source_term)}(?![A-Za-z0-9_])",
            target_term,
            text,
        )
    return text.replace(source_term, target_term)


def _unify_aliases(
    cues: list[_Cue],
    alias_groups: list[list[str]],
    term_base: dict[str, str],
    report: _Report,
) -> None:
    for group in alias_groups:
        members = [member for member in group if member]
        if len(members) < 2:
            continue
        canonical = next(
            (member for member in members if member in term_base.values()),
            None,
        )
        first: int | None = None
        for cue in cues:
            if cue.action != "translated" or not cue.parts:
                continue
            present = [member for member in members if member in " ".join(cue.parts)]
            if not present:
                continue
            if canonical is None:
                canonical, first = present[0], cue.dialogue_index
                continue
            for member in present:
                if member != canonical:
                    cue.parts = [part.replace(member, canonical) for part in cue.parts]
                    report.alias(
                        cue.dialogue_index, member, canonical,
                        f"unified to the first occurrence (cue #{first})",
                    )


def _split_parts(text: str, count: int, cjk: bool) -> list[str]:
    if count <= 1:
        return [text]
    parts: list[str] = []
    remaining = text
    total = len(text)
    for i in range(count - 1):
        target = total * (i + 1) // count
        cut = _cut_near(remaining, target, cjk)
        parts.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
        if not remaining:
            break
    if remaining:
        parts.append(remaining)
    return [part for part in parts if part]


def _wrap_translation(text: str, cjk: bool) -> list[str]:
    limit = _CJK_WRAP if cjk else _LATIN_WRAP
    if len(text) <= limit:
        return [text]
    lines: list[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = _cut_near(remaining, len(window), cjk)
        if cut <= 0:
            cut = limit
        lines.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
        if not remaining:
            break
    if remaining:
        lines.append(remaining)
    return lines


def _cut_near(text: str, target: int, cjk: bool) -> int:
    target = min(target, len(text))
    window = text[:target]
    if cjk:
        best = max([window.rfind(char) for char in _CJK_PUNCT] + [0])
        return best + 1 if best > 0 else target
    best = window.rfind(" ")
    return best + 1 if best > 0 else target


def _cue_tail(batch: list[_Cue]) -> list[tuple[str, str]]:
    tail: list[tuple[str, str]] = []
    for cue in batch:
        if cue.action == "translated" and cue.parts:
            tail.append((cue.source, " ".join(cue.parts)))
    return tail[-_MID_CONTEXT_SIZE:]


def _source_tail(batches: list[list[_Cue]], index: int) -> list[tuple[str, str]]:
    if index <= 0:
        return []
    return [
        (cue.source, "")
        for cue in batches[index - 1][-_MID_CONTEXT_SIZE:]
    ]


def _is_cjk(target_lang: str) -> bool:
    return target_lang.lower().split("-")[0] in _CJK_LANGS


def _build_output(
    normalized: NormalizedSubtitle, cues: list[_Cue], target_style: str
) -> NormalizedSubtitle:
    by_position = {cue.position: cue for cue in cues}
    dialogue = []
    for position, entry in enumerate(normalized.dialogue):
        cue = by_position.get(position)
        if cue is not None and cue.action == "translated" and cue.parts:
            lines = [
                line for line in entry.lines if line.style != target_style
            ]
            lines.extend(
                DialogueLine(style=target_style, content=part)
                for part in cue.parts
            )
            dialogue.append(entry.model_copy(update={"lines": lines}))
        else:
            dialogue.append(entry)
    return normalized.model_copy(update={"dialogue": dialogue})


def _sample(cues: list[_Cue], report: _Report) -> None:
    translated = [cue for cue in cues if cue.action == "translated" and cue.parts]
    if not translated:
        return
    count = max(1, math.ceil(len(translated) * _SAMPLE_MIN_RATIO))
    chosen = random.Random(0).sample(translated, count)
    report.sample(chosen)


def _notify(on_log: Callable[[str], None] | None, message: str) -> None:
    if on_log is not None:
        on_log(message)


class _Report:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._header: list[str] = []
        self._batches: list[str] = []
        self._lengths: list[str] = []
        self._terms: list[str] = []
        self._aliases: list[str] = []
        self._warnings: list[str] = []
        self._summary: list[str] = []
        self._sample: list[str] = []

    def _append(self, section: list[str], line: str) -> None:
        with self._lock:
            section.append(line)

    def header(self, path: Path) -> None:
        self._append(self._header, f"Translation report: {path}")

    def inputs(self, total: int, pending: int) -> None:
        self._append(
            self._header, f"Input: {total} cues ({pending} to translate)"
        )

    def batch(
        self,
        index: int,
        batch: list[_Cue],
        status: str,
        elapsed: float,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        first = batch[0].dialogue_index
        last = batch[-1].dialogue_index
        self._append(
            self._batches,
            f"  batch {index}: cues #{first}..#{last}, {status}, "
            f"{elapsed:.1f}s, prompt {prompt_tokens}, "
            f"completion {completion_tokens}",
        )

    def length(self, index: int, source_len: int, target_len: int, detail: str) -> None:
        self._append(
            self._lengths,
            f"  cue #{index}: source {source_len} chars, "
            f"translation {target_len} chars, {detail}",
        )

    def term(self, index: int, source: str, target: str) -> None:
        self._append(
            self._terms, f"  cue #{index}: '{source}' -> '{target}' (term base)"
        )

    def alias(self, index: int, before: str, after: str, reason: str) -> None:
        self._append(
            self._aliases,
            f"  cue #{index}: '{before}' -> '{after}' ({reason})",
        )

    def warning(self, message: str) -> None:
        self._append(self._warnings, f"  ! WARNING: {message}")

    def record(self, message: str) -> None:
        self._append(self._warnings, f"  - {message}")

    def summary(self, cues: list[_Cue], tokens: _TokenCount) -> None:
        translated = sum(1 for cue in cues if cue.action == "translated")
        skipped = sum(1 for cue in cues if cue.skip)
        failed = sum(1 for cue in cues if cue.action == "failed")
        kept = len(cues) - translated - skipped - failed
        self._append(
            self._summary,
            f"Summary: {len(cues)} cues, translated {translated}, "
            f"skipped {skipped}, failed {failed}, kept old {kept}",
        )

    def tokens(self, tokens: _TokenCount) -> None:
        self._append(
            self._summary,
            f"Tokens: {tokens.requests} requests, "
            f"prompt {tokens.prompt}, completion {tokens.completion}",
        )

    def api_time(self, total_elapsed: float) -> None:
        self._append(
            self._summary, f"API time: {total_elapsed:.1f}s"
        )

    def sample(self, chosen: list[_Cue]) -> None:
        self._append(
            self._sample,
            f"Sample review ({len(chosen)} of translated cues):",
        )
        for cue in chosen:
            self._append(self._sample, f"  cue #{cue.dialogue_index}")
            self._append(self._sample, f"    source: {cue.source}")
            self._append(
                self._sample, f"    translation: {' '.join(cue.parts)}"
            )

    def render(self) -> str:
        sections = [
            self._header,
            self._batches,
            self._lengths,
            self._terms,
            self._aliases,
            self._warnings,
            self._summary,
            self._sample,
        ]
        return "\n".join(
            "\n".join(section) for section in sections if section
        )
