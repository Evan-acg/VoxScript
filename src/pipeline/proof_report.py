from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReportSummary:
    total_lines: int = 0
    timeline_issues: int = 0
    missing_lines: int = 0
    extra_lines: int = 0
    translation_issues: int | None = None
    cache_hits: list[str] = field(default_factory=list)
    elapsed: float = 0.0


@dataclass
class TimelineIssue:
    line_index: int
    text: str
    original_start: float
    aligned_start: float
    original_end: float
    aligned_end: float
    offset_start: float
    offset_end: float
    severity: str = "warning"


@dataclass
class ExtraLineInfo:
    line_index: int
    text: str
    confidence: float


@dataclass
class MissingLineInfo:
    start: float
    end: float
    duration: float
    suggested_text: str | None = None


@dataclass
class TranslationIssue:
    line_index: int
    original_text: str
    translated_text: str
    suggestion: str | None = None
    score: float | None = None


@dataclass
class ExecutionInfo:
    phases: dict[str, float] = field(default_factory=dict)
    cache_hits: list[str] = field(default_factory=list)


@dataclass
class ProofReport:
    summary: ReportSummary = field(default_factory=ReportSummary)
    timeline_issues: list[TimelineIssue] = field(default_factory=list)
    suspected_extra_lines: list[ExtraLineInfo] = field(default_factory=list)
    suspected_missing_lines: list[MissingLineInfo] = field(default_factory=list)
    translation_issues: list[TranslationIssue] | None = None
    execution_info: ExecutionInfo = field(default_factory=ExecutionInfo)


def _box_line(text: str, width: int = 38) -> str:
    text = text.ljust(width)
    return f"║ {text} ║"


def format_terminal_summary(report: ProofReport) -> str:
    s = report.summary
    lines = [
        "╔══════════════════════════════════════════╗",
        "║              校对报告摘要                 ║",
        "╠══════════════════════════════════════════╣",
        _box_line(f"总行数:             {s.total_lines}"),
        _box_line(f"时间轴偏移:         {s.timeline_issues} 处"),
        _box_line(f"可能多余台词:       {s.extra_lines} 处"),
        _box_line(f"可能缺失台词:       {s.missing_lines} 处"),
    ]
    if s.translation_issues is not None:
        lines.append(_box_line(f"翻译问题:           {s.translation_issues} 处"))
    if s.cache_hits:
        lines.append(_box_line(f"缓存命中:           {', '.join(s.cache_hits)}"))
    lines.append(_box_line(f"耗时:               {s.elapsed:.1f}s"))
    lines.append("╚══════════════════════════════════════════╝")

    if report.timeline_issues:
        lines.append("")
        lines.append("--- 时间轴偏移详情 ---")
        for issue in report.timeline_issues[:10]:
            lines.append(
                f"  #{issue.line_index} 偏移 {issue.offset_start:+.2f}s/{issue.offset_end:+.2f}s "
                f"({issue.original_start:.1f}->{issue.aligned_start:.1f}) [{issue.severity}]"
            )
        if len(report.timeline_issues) > 10:
            lines.append(f"  ... 还有 {len(report.timeline_issues) - 10} 处")

    if report.suspected_missing_lines:
        lines.append("")
        lines.append("--- 可能缺失台词 ---")
        for m in report.suspected_missing_lines:
            gap_text = f"  [{m.start:.1f}s - {m.end:.1f}s] ({m.duration:.1f}s)"
            if m.suggested_text:
                gap_text += f" → \"{m.suggested_text}\""
            lines.append(gap_text)

    if report.suspected_extra_lines:
        lines.append("")
        lines.append("--- 可能多余台词 ---")
        for e in report.suspected_extra_lines[:10]:
            lines.append(f"  #{e.line_index} conf={e.confidence:.2f} \"{e.text[:60]}\"")
        if len(report.suspected_extra_lines) > 10:
            lines.append(f"  ... 还有 {len(report.suspected_extra_lines) - 10} 处")

    return "\n".join(lines)


def format_json_report(report: ProofReport) -> str:
    def _as_dict(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _as_dict(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, list):
            return [_as_dict(i) for i in obj]
        if isinstance(obj, float):
            return round(obj, 4)
        return obj

    return json.dumps(_as_dict(report), ensure_ascii=False, indent=2)
