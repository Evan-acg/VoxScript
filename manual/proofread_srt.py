"""
proofread_srt.py — 逐条校对译文 SRT，输出修正版（不修改原始文件）

用法:
    python proofread_srt.py <原文.srt> <译文.srt> [输出.srt]

修正规则（按条目号）定义在 CORRECTIONS 字典中。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def parse_srt(filepath: str) -> dict[int, tuple[str, str, str]]:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    blocks = re.split(r'\n\n+', content.strip())
    entries: dict[int, tuple[str, str, str]] = {}
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        idx = int(lines[0].strip())
        time_range = lines[1].strip()
        text = '\n'.join(lines[2:])
        start, end = time_range.split(' --> ')
        entries[idx] = (start, end, text)
    return entries


def write_srt(entries: dict[int, tuple[str, str, str]], filepath: str) -> None:
    lines: list[str] = []
    for idx in sorted(entries):
        start, end, text = entries[idx]
        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append('')
    Path(filepath).write_text('\n'.join(lines), encoding='utf-8')


def escape_ass(text: str) -> str:
    text = text.replace('\\', '\\\\')
    text = text.replace('{', '\\{')
    text = text.replace('}', '\\}')
    return text


# ── 逐条修正表 ──────────────────────────────────────────────
# key=条目号, value=修正后的中文文本（不修改时间戳/序号）
CORRECTIONS: dict[int, str] = {
}


# ── 模式修正（不依赖条目号） ──────────────────────────────
def pattern_fixes(text: str) -> str:
    """对文本应用全局模式修正。"""
    # 引号统一（全角/半角/弯引号 -> 标准直角引号）
    text = re.sub(r' +', ' ', text)
    return text
    return text


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    src_path = Path(sys.argv[1])
    tgt_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else (
        tgt_path.with_stem(tgt_path.stem + '-corrected'))

    originals = parse_srt(str(src_path))
    translations = parse_srt(str(tgt_path))

    if not originals:
        print(f"Error: no entries found in {src_path}")
        sys.exit(1)

    indices = sorted(set(originals) | set(translations))
    corrected: dict[int, tuple[str, str, str]] = {}

    for idx in indices:
        trans = translations.get(idx)
        if not trans:
            continue

        start, end, text = trans

        if idx in CORRECTIONS:
            text = CORRECTIONS[idx]
        else:
            text = pattern_fixes(text)

        corrected[idx] = (start, end, text)

    write_srt(corrected, str(out_path))
    print(f"OK -> {out_path} ({len(corrected)} entries)")


if __name__ == "__main__":
    main()
