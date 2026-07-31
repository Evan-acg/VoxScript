"""
merge_srt_to_ass.py — 通用双语字幕合并工具

将原文 SRT 与译文 SRT 合并为 ASS 格式，不含任何字幕数据。

用法:
    python merge_srt_to_ass.py <原文.srt> <译文.srt> [输出.ass]

约定:
    - 译文文本以 "##注释" 开头 → 使用 注释 样式（顶部居中）
    - 其余条目使用 Default + Original 样式（底部双语）
"""

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


def srt_time_to_ass(t: str) -> str:
    t = t.replace(',', '.')
    parts = t.split(':')
    hours = str(int(parts[0]))
    sec_parts = parts[2].split('.')
    seconds = sec_parts[0]
    cs = sec_parts[1][:2].ljust(2, '0') if len(sec_parts) > 1 else '00'
    return f"{hours}:{parts[1]}:{seconds}.{cs}"


def escape_ass(text: str) -> str:
    text = text.replace('\\', '\\\\')
    text = text.replace('{', '\\{')
    text = text.replace('}', '\\}')
    return text


def is_blank(text: str) -> bool:
    return not text.strip() or set(text.strip()) <= {'\u200b', '\u200c', '\u200d', '\ufeff'}


def gen_dialogue(style: str, start: str, end: str, text: str) -> str:
    ass_start = srt_time_to_ass(start)
    ass_end = srt_time_to_ass(end)
    return f"Dialogue: 2,{ass_start},{ass_end},{style},,0,0,0,,{text}"


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    src_path = Path(sys.argv[1])
    tgt_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else src_path.with_suffix('.ass')

    originals = parse_srt(str(src_path))
    translations = parse_srt(str(tgt_path))

    if not originals:
        print(f"Error: no entries found in {src_path}")
        sys.exit(1)

    indices = sorted(set(originals) | set(translations))

    out_lines: list[str] = [
        "[Script Info]",
        "Title: Bilingual Subtitles",
        "ScriptType: v4.00+",
        "Collisions: Normal",
        "PlayDepth: 0",
        "",
        STYLES.strip(),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    for idx in indices:
        orig = originals.get(idx)
        trans = translations.get(idx)

        if not orig:
            print(f"Warning: index {idx} missing in original, skipped")
            continue

        start, end, orig_text = orig
        trans_text = trans[2] if trans else ""

        trans_escaped = escape_ass(trans_text.replace('\n', ' '))
        orig_escaped = escape_ass(orig_text.replace('\n', ' '))

        if trans_text.startswith("##注释"):
            clean_trans = trans_escaped.removeprefix("##注释").lstrip()
            text = f"{{\\fad(120,120)}}{clean_trans}"
            style = "注释"
        elif not is_blank(trans_escaped) and not is_blank(orig_escaped):
            text = f"{{\\fad(120,120)}}{trans_escaped}\\N{{\\rOriginal}}{orig_escaped}"
            style = "Default"
        elif not is_blank(trans_escaped):
            text = f"{{\\fad(120,120)}}{trans_escaped}"
            style = "Default"
        elif not is_blank(orig_escaped):
            text = f"{{\\fad(120,120)}}{orig_escaped}"
            style = "Default"
        else:
            continue

        out_lines.append(gen_dialogue(style, start, end, text))

    out_path.write_text('\n'.join(out_lines) + '\n', encoding='utf-8')
    print(f"OK {out_path} ({len(indices)} entries)")


if __name__ == "__main__":
    main()
