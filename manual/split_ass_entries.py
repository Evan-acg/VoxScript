"""
split_ass_entries.py — 拆分双语 ASS 中的超长条目

从双语 ASS 提取原文和译文，英文按自然断句拆分（≤70 字符），
中文按比例切分并回退到最近标点（≤40 字符），输出两个 SRT。

用法:
    python split_ass_entries.py <双语.ass> <原文.srt> <译文.srt>

与 merge_srt_to_ass.py 配合修复超长条目：
    python split_ass_entries.py out.ass original.srt translated.srt
    python merge_srt_to_ass.py original.srt translated.srt out.ass
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ASS_DIALOGUE_RE = re.compile(
    r'Dialogue:\s*(\d+),'
    r'(\d+:\d+:\d+\.\d+),'
    r'(\d+:\d+:\d+\.\d+),'
    r'([^,]*),'
    r'([^,]*),'
    r'(\d+),'
    r'(\d+),'
    r'(\d+),'
    r'([^,]*),'
    r'(.+)'
)

EN_MAX = 70
CN_MAX = 40


def time_to_ms(t: str) -> int:
    parts = t.split(':')
    s_parts = parts[2].split('.')
    return int(parts[0]) * 3600000 + int(parts[1]) * 60000 + int(s_parts[0]) * 1000 + int(s_parts[1]) * 10


def ms_to_srt(ms: int) -> str:
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_bilingual_ass(path: str) -> list[dict]:
    entries, in_events = [], False
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            ls = line.strip()
            if ls == '[Events]':
                in_events = True; continue
            if in_events and ls.startswith('['):
                in_events = False; continue
            if not in_events or not ls.startswith('Dialogue:'):
                continue
            m = ASS_DIALOGUE_RE.match(ls)
            if not m:
                continue
            text = re.sub(r'\{[^}]*\}', '', m.group(10))
            cn, en = '', ''
            if r'\N' in text:
                parts = text.split(r'\N', 1)
                cn = parts[0].strip()
                en = re.sub(r'\\rOriginal\s*', '', parts[1]).strip()
            else:
                cn = text.strip()
            entries.append({'start': m.group(2), 'end': m.group(3), 'cn': cn, 'en': en})
    return entries


def refine_back(text: str, pos: int) -> int:
    """Adjust position backward to nearest boundary (for CN)."""
    if pos <= 0 or pos >= len(text):
        return pos
    start = max(0, pos - 10)
    for p in range(pos - 1, start - 1, -1):
        if text[p] in '.!?。！？,;，； ' and pos - (p + 1) <= 5:
            return p + 1
    return pos


def split_en(text: str, max_len: int = EN_MAX) -> list[str]:
    """Split English text at natural boundaries."""
    if not text or len(text) <= max_len:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        best = -1
        for sep in ['. ', '! ', '? ']:
            p = text.rfind(sep, 0, max_len)
            if p > best: best = p + len(sep)
        if best < 0:
            for sep in [', ', '; ', ',', ';']:
                p = text.rfind(sep, 0, max_len)
                if p > best: best = p + len(sep)
        if best < 0:
            p = text.rfind(' ', max(max_len - 20, 0), max_len)
            if p > 0: best = p + 1
        if best <= 0:
            p = text.find(' ', max_len)
            best = (p + 1) if p > 0 else max_len
        chunks.append(text[:best].strip())
        text = text[best:].strip()
    return chunks


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    entries = parse_bilingual_ass(str(Path(sys.argv[1])))
    out_orig = Path(sys.argv[2])
    out_trans = Path(sys.argv[3])
    print(f"  Total entries: {len(entries)}")

    new_entries: list[tuple[int, str, str, str, str]] = []
    split_count = 0

    for e in entries:
        start_ms = time_to_ms(e['start'])
        end_ms = time_to_ms(e['end'])
        duration = end_ms - start_ms
        cn, en = e['cn'], e['en']

        if len(en or '') <= EN_MAX and len(cn or '') <= CN_MAX:
            new_entries.append((start_ms, e['start'], e['end'], cn, en))
            continue

        split_count += 1

        en_segs = split_en(en, EN_MAX) if en else ['']
        n = len(en_segs)

        if cn:
            cn_len = len(cn)
            cuts = {0, cn_len}
            for j in range(1, n):
                cuts.add(refine_back(cn, int(j * cn_len / n)))
            cuts = sorted(cuts)
            cn_segs = [cn[cuts[i]:cuts[i+1]].strip() for i in range(len(cuts)-1)]
        else:
            cn_segs = [''] * n

        while len(cn_segs) < n:
            cn_segs.append('')
        cn_segs = cn_segs[:n]

        weights = [max(len(en_segs[j]) + len(cn_segs[j]), 1) for j in range(n)]
        total_w = sum(weights)

        allocated = 0
        for j in range(n):
            dur = duration * weights[j] // total_w if j < n - 1 else duration - allocated
            dur = max(dur, 200)
            ss = ms_to_srt(start_ms + allocated)
            se = ms_to_srt(start_ms + allocated + dur)
            new_entries.append((start_ms + allocated, ss, se, cn_segs[j], en_segs[j]))
            allocated += dur

        if allocated != duration:
            last = new_entries[-1]
            new_entries[-1] = (last[0], last[1], ms_to_srt(end_ms), last[3], last[4])

    new_entries.sort(key=lambda x: x[0])

    orig_lines, trans_lines = [], []
    for idx, (_, ss, se, cn, en) in enumerate(new_entries, 1):
        orig_lines += [str(idx), f"{ss} --> {se}", en if en else cn, '']
        trans_lines += [str(idx), f"{ss} --> {se}", cn if cn else en, '']

    out_orig.write_text('\n'.join(orig_lines), encoding='utf-8')
    out_trans.write_text('\n'.join(trans_lines), encoding='utf-8')

    print(f"  Split: {split_count} entries -> {len(new_entries)} total")
    print(f"  OK {out_orig}")
    print(f"  OK {out_trans}")


if __name__ == '__main__':
    main()
