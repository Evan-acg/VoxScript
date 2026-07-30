"""
merge_ass_to_ass.py — 合并两个同源 ASS 为双语 ASS

将原文 ASS 与译文 ASS 合并为双语 ASS 格式。
两个文件应来自同一来源（相同时间轴结构）。

用法:
    python merge_ass_to_ass.py <原文.ass> <译文.ass> [输出.ass]

约定:
    - staff 样式的条目 → 使用 注释 样式（顶部居中）
    - 其余条目使用 Default + Original 样式（底部双语）
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


STYLES = """\
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Microsoft YaHei,11,&H00F2F2F2,&H0000FFFF,&H000F0F0F,&H80000000,0,0,0,0,100,100,0.6,0,1,1,1.2,2,20,20,5,134
Style: Original,Arial,12,&H00D7E3E8,&H0000FFFF,&H00222222,&H00000000,0,0,0,0,100,100,0.2,0,1,1,1,2,20,20,11,1
Style: 注释,微软雅黑,18,&H00FFFFFF,&HFF000000,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0.5,8,0,0,5,1
"""

HEADER = """\
[Script Info]
Title: Bilingual Subtitles
ScriptType: v4.00+
Collisions: Normal
PlayResX: 848
PlayResY: 480
Timer: 100.0000
"""


def read_file(path: str | Path) -> str:
    path = Path(path)
    for enc in ('utf-8', 'utf-16-le', 'shift-jis', 'gbk'):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            pass
    return path.read_text(encoding='ansi')


ASS_TIME_RE = re.compile(r'Dialogue:\s*(\d+),([^,]+),([^,]+),([^,]+),([^,]+),(\d+),(\d+),(\d+),([^,]*),(.+)')


def parse_events(content: str) -> list[dict]:
    entries: list[dict] = []
    in_events = False

    for line in content.splitlines():
        line = line.strip()
        if line == '[Events]':
            in_events = True
            continue
        if in_events and line.startswith('['):
            in_events = False
            continue
        if not in_events:
            continue
        if not line.startswith('Dialogue:'):
            continue

        m = ASS_TIME_RE.match(line)
        if not m:
            continue

        entries.append({
            'layer': int(m.group(1)),
            'start': m.group(2),
            'end': m.group(3),
            'style': m.group(4),
            'actor': m.group(5),
            'margin_l': int(m.group(6)),
            'margin_r': int(m.group(7)),
            'margin_v': int(m.group(8)),
            'effect': m.group(9),
            'text': m.group(10),
        })

    return entries


def time_to_ms(t: str) -> int:
    parts = t.split(':')
    h = int(parts[0])
    m = int(parts[1])
    sec_parts = parts[2].split('.')
    s = int(sec_parts[0])
    cs = int(sec_parts[1]) if len(sec_parts) > 1 else 0
    return h * 3600000 + m * 60000 + s * 1000 + cs * 10


def clean_text(text: str) -> str:
    text = re.sub(r'\{[^}]*\}', '', text)
    text = text.replace('\\N', ' ')
    text = text.replace('\\n', ' ')
    text = ' '.join(text.split())
    return text


def escape_ass(text: str) -> str:
    text = text.replace('\\', '\\\\')
    text = text.replace('{', '\\{')
    text = text.replace('}', '\\}')
    return text


TOLERANCE_MS = 300


def match_entries(
    src: list[dict],
    tgt: list[dict],
) -> list[tuple[dict | None, dict | None]]:
    matched: list[tuple[dict | None, dict | None]] = []
    tgt_used: set[int] = set()

    for se in src:
        s_ms = time_to_ms(se['start'])
        best: int | None = None
        best_diff = TOLERANCE_MS + 1
        for i, te in enumerate(tgt):
            if i in tgt_used:
                continue
            diff = abs(s_ms - time_to_ms(te['start']))
            if diff < best_diff:
                best_diff = diff
                best = i
        if best is not None:
            tgt_used.add(best)
            matched.append((se, tgt[best]))
        else:
            matched.append((se, None))

    for i, te in enumerate(tgt):
        if i not in tgt_used:
            matched.append((None, te))

    matched.sort(key=lambda pair: (
        time_to_ms(pair[0]['start']) if pair[0] else time_to_ms(pair[1]['start'])
    ))
    return matched


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    src_path = Path(sys.argv[1])
    tgt_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else (
        tgt_path.with_name(tgt_path.stem + '.bilingual.ass')
    )

    src_entries = parse_events(read_file(src_path))
    tgt_entries = parse_events(read_file(tgt_path))

    if not src_entries:
        print(f"Error: no Dialogue entries found in {src_path}")
        sys.exit(1)
    if not tgt_entries:
        print(f"Error: no Dialogue entries found in {tgt_path}")
        sys.exit(1)

    print(f"  Source entries: {len(src_entries)}")
    print(f"  Target entries: {len(tgt_entries)}")

    matched = match_entries(src_entries, tgt_entries)
    unmatched = sum(1 for s, t in matched if s is None or t is None)
    if unmatched:
        print(f"  Warning: {unmatched} unmatched entries")

    out_lines: list[str] = [
        HEADER.strip(),
        '',
        STYLES.strip(),
        '',
        '[Events]',
        'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text',
    ]

    for se, te in matched:
        start = (se or te)['start']
        end = (se or te)['end']

        src_text = clean_text(se['text']) if se else ''
        tgt_text = clean_text(te['text']) if te else ''

        src_esc = escape_ass(src_text) if src_text else ''
        tgt_esc = escape_ass(tgt_text) if tgt_text else ''

        is_staff = (se and se['style'] == 'staff') or (te and te['style'] == 'staff')

        if is_staff:
            display_text = tgt_esc if tgt_esc else src_esc
            out_lines.append(
                f"Dialogue: 2,{start},{end},注释,,0,0,0,,{{\\fad(120,120)}}{display_text}"
            )
        else:
            if not tgt_esc:
                display_text = f"{{\\fad(120,120)}}{src_esc}"
            elif not src_esc:
                display_text = f"{{\\fad(120,120)}}{tgt_esc}"
            else:
                display_text = f"{{\\fad(120,120)}}{tgt_esc}\\N{{\\rOriginal}}{src_esc}"
            out_lines.append(
                f"Dialogue: 2,{start},{end},Default,,0,0,0,,{display_text}"
            )

    out_path.write_text('\n'.join(out_lines) + '\n', encoding='utf-8')
    print(f"OK {out_path} ({len(matched)} entries)")


if __name__ == '__main__':
    main()
