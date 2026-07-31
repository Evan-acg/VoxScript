"""
cross_validate_ass_srt.py — 交叉验证 ASS + SRT，输出配对 SRT 文件

从现有 ASS（UTF-16 LE）提取中文翻译，从 SRT 提取日语原文，
按时间戳匹配交叉验证，输出一对用于 merge_srt_to_ass.py 的 SRT。

用法:
    python cross_validate_ass_srt.py <ass.ass> <srt.srt> [原文.srt] [译文.srt]

步骤:
    1. 解析 ASS 和 SRT
    2. 按时间戳匹配条目
    3. 输出交叉验证报告
    4. 输出一对 SRT（原文 + 译文）

样式约定:
    *Default            → 中文对白（译文轨道）
    OPCN/EDCN           → 中文歌词（译文轨道）
    OPJP/EDJP           → 日语歌词（原文轨道）
    STAFF/Bann/Suzu 等  → ##注释（译文轨道，注释样式）
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def read_file(path: str | Path) -> str:
    path = Path(path)
    for enc in ('utf-16-le', 'utf-8', 'gbk', 'shift-jis'):
        try:
            raw = path.read_bytes()
            if raw[:2] == b'\xff\xfe':
                return raw.decode('utf-16-le')
            return raw.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return path.read_text(encoding='ansi')


ASS_DIALOGUE_RE = re.compile(
    r'Dialogue:\s*(\d+),'
    r'([^,]+),'
    r'([^,]+),'
    r'([^,]*),'
    r'([^,]*),'
    r'(\d+),'
    r'(\d+),'
    r'(\d+),'
    r'([^,]*),'
    r'(.+)'
)

ANNOTATION_STYLES = {'STAFF', 'Bann', 'Bann2', 'Suzu', 'Suzu2', 'frz', 'frz_x', 'title', 'yu', 'CK', 'fad1'}
LYRICS_JP_STYLES = {'OPJP', 'EDJP'}
LYRICS_CN_STYLES = {'OPCN', 'EDCN'}
DIALOG_STYLE = {'*Default', 'Default'}


def parse_ass(path: str) -> list[dict]:
    content = read_file(path)
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
        if not in_events or not line.startswith('Dialogue:'):
            continue

        m = ASS_DIALOGUE_RE.match(line)
        if not m:
            continue

        text_raw = m.group(10)
        text_clean = re.sub(r'\{[^}]*\}', '', text_raw).replace('\r', '').strip()

        style = m.group(4).strip()

        entries.append({
            'start': m.group(2),
            'end': m.group(3),
            'style': style,
            'text_raw': text_raw,
            'text_clean': text_clean,
        })

    return entries


def parse_srt(path: str) -> list[dict]:
    content = Path(path).read_text(encoding='utf-8')
    blocks = re.split(r'\n\n+', content.strip())
    entries: list[dict] = []

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        idx = lines[0].strip()
        time_range = lines[1].strip()
        text = '\n'.join(lines[2:])
        start, end = time_range.split(' --> ')
        entries.append({
            'idx': int(idx),
            'start': start,
            'end': end,
            'text': text,
        })

    return entries


def time_to_ms(t: str) -> int:
    parts = t.split(':')
    sec_parts = parts[2].replace(',', '.').split('.')
    s = int(sec_parts[0])
    cs = int(sec_parts[1]) if len(sec_parts) > 1 else 0
    return int(parts[0]) * 3600000 + int(parts[1]) * 60000 + s * 1000 + cs * 10


def ms_to_srt(ms: int) -> str:
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_time_for_display(t: str) -> str:
    """Normalize time format for display: HH:MM:SS.mmm"""
    t = t.replace(',', '.')
    parts = t.split(':')
    sec_parts = parts[2].split('.')
    sec = sec_parts[0]
    ms = (sec_parts[1] + '000')[:3] if len(sec_parts) > 1 else '000'
    return f"{int(parts[0]):02d}:{parts[1]}:{sec}.{ms}"


def normalize_ts(t: str) -> str:
    """Convert any time format to SRT: HH:MM:SS,mmm"""
    t = t.replace(',', '.')
    parts = t.split(':')
    sec_parts = parts[2].split('.')
    ms = sec_parts[1].ljust(3, '0')[:3] if len(sec_parts) > 1 else '000'
    return f"{int(parts[0]):02d}:{int(parts[1]):02d}:{int(sec_parts[0]):02d},{ms}"


def match_entries(
    srt_entries: list[dict],
    ass_dialog: list[dict],
) -> list[dict]:
    """Match SRT entries to ASS *Default entries by time overlap."""
    results: list[dict] = []
    srt_by_start = sorted(srt_entries, key=lambda e: time_to_ms(e['start']))
    ass_by_start = sorted(ass_dialog, key=lambda e: time_to_ms(e['start']))

    ass_used: set[int] = set()

    for srt_e in srt_by_start:
        s_start = time_to_ms(srt_e['start'])
        s_end = time_to_ms(srt_e['end'])

        overlapping: list[tuple[int, dict]] = []
        for i, ass_e in enumerate(ass_by_start):
            a_start = time_to_ms(ass_e['start'])
            a_end = time_to_ms(ass_e['end'])

            overlap_start = max(s_start, a_start)
            overlap_end = min(s_end, a_end)

            if overlap_start < overlap_end:
                overlap_ms = overlap_end - overlap_start
                overlapping.append((i, ass_e, overlap_ms))

        overlapping.sort(key=lambda x: x[2], reverse=True)
        matched: list[int] = []
        matched_entries: list[dict] = []

        for i, ass_e, oms in overlapping:
            if i not in ass_used:
                ass_used.add(i)
                matched.append(i)
                matched_entries.append(ass_e)

        matched_entries.sort(key=lambda e: time_to_ms(e['start']))

        results.append({
            'srt': srt_e,
            'ass_indices': matched,
            'ass_entries': matched_entries,
        })

    unmatched_ass = [
        (i, e) for i, e in enumerate(ass_by_start) if i not in ass_used
    ]

    return results, unmatched_ass, ass_by_start


def split_jp_text(text: str, n: int, durations: list[int]) -> list[str]:
    """Split Japanese text into n parts proportional to durations."""
    if n <= 0:
        return []
    if n == 1:
        return [text]

    segments = re.split(r'(?<=[。！？.!?\n])', text)
    segments = [s.strip() for s in segments if s.strip()]

    if not segments:
        total = sum(durations)
        ratios = [d / total for d in durations] if total > 0 else [1 / n] * n
        result = []
        pos = 0
        for r in ratios:
            chunk_len = max(1, int(len(text) * r))
            chunk_len = min(chunk_len, len(text) - pos)
            result.append(text[pos:pos + chunk_len].strip())
            pos += chunk_len
            if pos >= len(text):
                break
        while len(result) < n:
            result.append('')
        return result[:n]

    total_dur = sum(durations)
    result: list[str] = []
    seg_pos = 0
    allocated_dur = 0

    for j in range(n):
        target_dur = durations[j]
        target_ratio = target_dur / total_dur if total_dur > 0 else 1 / n
        target_segs = max(1, int(len(segments) * target_ratio))

        if j == n - 1:
            chunk = segments[seg_pos:]
        else:
            chunk = segments[seg_pos:seg_pos + target_segs]

        result.append(''.join(chunk).strip())
        seg_pos += len(chunk)
        allocated_dur += target_dur

    while len(result) < n:
        result.append('')
    return result[:n]


def output_srt(
    entries: list[tuple[int, str, str, str]],
    path: str,
) -> None:
    lines: list[str] = []
    for idx, start, end, text in entries:
        text = text if text.strip() else '\u200b'  # zero-width space to keep 3-line SRT format
        lines.extend([
            str(idx),
            f"{normalize_ts(start)} --> {normalize_ts(end)}",
            text,
            '',
        ])
    Path(path).write_text('\n'.join(lines), encoding='utf-8')
    print(f"  OK {path} ({len(entries)} entries)")


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    ass_path = Path(sys.argv[1])
    srt_path = Path(sys.argv[2])
    out_orig = Path(sys.argv[3]) if len(sys.argv) > 3 else ass_path.with_name(ass_path.stem + '-original.srt')
    out_trans = Path(sys.argv[4]) if len(sys.argv) > 4 else ass_path.with_name(ass_path.stem + '-translated.srt')

    # 1. Parse
    print("Parsing ASS...")
    all_ass = parse_ass(str(ass_path))
    print(f"  ASS total entries: {len(all_ass)}")

    print("Parsing SRT...")
    srt_entries = parse_srt(str(srt_path))
    print(f"  SRT total entries: {len(srt_entries)}")

    # 2. Classify ASS entries
    dialog_cn = [e for e in all_ass if e['style'] in DIALOG_STYLE]
    lyrics_jp = [e for e in all_ass if e['style'] in LYRICS_JP_STYLES]
    lyrics_cn = [e for e in all_ass if e['style'] in LYRICS_CN_STYLES]
    annotations = [e for e in all_ass if e['style'] in ANNOTATION_STYLES]
    other = [e for e in all_ass if e['style'] not in DIALOG_STYLE | LYRICS_JP_STYLES | LYRICS_CN_STYLES | ANNOTATION_STYLES]

    print(f"\nASS分类:")
    print(f"  对白 (*Default/Default): {len(dialog_cn)}")
    print(f"  日语歌词 (OPJP/EDJP): {len(lyrics_jp)}")
    print(f"  中文歌词 (OPCN/EDCN): {len(lyrics_cn)}")
    print(f"  注释/制作组: {len(annotations)}")
    print(f"  其他: {len(other)}")

    # 3. Match entries
    print("\n交叉验证对白条目...")
    match_results, unmatched_ass, ass_by_start = match_entries(srt_entries, dialog_cn)

    # 4. Print report
    print(f"\n{'='*60}")
    print(f"交叉验证报告")
    print(f"{'='*60}")

    matched_srt = 0
    matched_ass_count = 0

    for result in match_results:
        if result['ass_entries']:
            matched_srt += 1
            matched_ass_count += len(result['ass_entries'])

    extra_ass = len(unmatched_ass)
    extra_srt_count = sum(1 for r in match_results if not r['ass_entries'])

    print(f"\nSRT条目总数: {len(srt_entries)}")
    print(f"  已匹配到ASS对白: {matched_srt}")
    print(f"  未匹配到ASS对白: {extra_srt_count}")
    print(f"\nASS对白条目总数: {len(dialog_cn)}")
    print(f"  已匹配到SRT: {matched_ass_count}")
    print(f"  未匹配到SRT: {extra_ass}")

    print(f"\n--- 详细匹配 ---")
    for i, result in enumerate(match_results):
        srt_e = result['srt']
        s_start_d = format_time_for_display(srt_e['start'])
        s_end_d = format_time_for_display(srt_e['end'])
        s_text = srt_e['text'].replace('\n', ' ')[:60]

        if result['ass_entries']:
            ass_texts = [e['text_clean'][:40] for e in result['ass_entries']]
            print(f"\nSRT #{srt_e['idx']}: {s_start_d} -> {s_end_d}")
            print(f"  JP: {s_text}")
            for j, (ass_e, at) in enumerate(zip(result['ass_entries'], ass_texts)):
                a_start_d = format_time_for_display(ass_e['start'])
                a_end_d = format_time_for_display(ass_e['end'])
                print(f"  ASS #{j+1}: {a_start_d} -> {a_end_d} | CN: {at}")
        else:
            print(f"\nSRT #{srt_e['idx']}: {s_start_d} -> {s_end_d}")
            print(f"  JP: {s_text}")
            print(f"  [!] 无匹配ASS条目")

    if unmatched_ass:
        print(f"\n--- 未匹配的ASS对白 ({len(unmatched_ass)}条) ---")
        for i, e in unmatched_ass[:10]:
            d_start = format_time_for_display(e['start'])
            d_end = format_time_for_display(e['end'])
            print(f"  #{i}: {d_start} -> {d_end} | {e['text_clean'][:60]}")
        if len(unmatched_ass) > 10:
            print(f"  ... 还有{len(unmatched_ass) - 10}条")

    # 5. Generate output SRT files
    print(f"\n{'='*60}")
    print(f"生成配对SRT文件")
    print(f"{'='*60}")

    pairs: list[tuple[str, str, str, str]] = []

    # 5a. Matched dialog entries
    for result in match_results:
        srt_e = result['srt']
        s_start = srt_e['start']
        s_end = srt_e['end']
        s_text = srt_e['text']

        if not result['ass_entries']:
            pairs.append((s_start, s_end, s_text, ''))
            continue

        ass_entries = result['ass_entries']

        if len(ass_entries) == 1:
            pairs.append((s_start, s_end, s_text, ass_entries[0]['text_clean']))
        else:
            total_dur = time_to_ms(s_end) - time_to_ms(s_start)
            durations = []
            for ae in ass_entries:
                d = time_to_ms(ae['end']) - time_to_ms(ae['start'])
                durations.append(max(d, 100))

            jp_parts = split_jp_text(s_text, len(ass_entries), durations)

            allocated = 0
            for j, ae in enumerate(ass_entries):
                dur = durations[j]
                part_start = time_to_ms(s_start) + allocated
                part_end = min(part_start + dur, time_to_ms(s_end))
                if j == len(ass_entries) - 1:
                    part_end = time_to_ms(s_end)

                start_str = ms_to_srt(part_start)
                end_str = ms_to_srt(part_end)
                jp_text = jp_parts[j] if j < len(jp_parts) else ''

                pairs.append((start_str, end_str, jp_text, ae['text_clean']))
                allocated += dur

    # 5b. Unmatched ASS dialog entries
    for i, ae in unmatched_ass:
        pairs.append((ae['start'], ae['end'], '', ae['text_clean']))

    # 5c. OP/ED lyrics (merge JP and CN by timestamp)
    jp_lyrics_map: dict[str, list[dict]] = {}
    for le in lyrics_jp:
        key = f"{time_to_ms(le['start'])}_{time_to_ms(le['end'])}"
        jp_lyrics_map.setdefault(key, []).append(le)

    cn_lyrics_map: dict[str, list[dict]] = {}
    for le in lyrics_cn:
        key = f"{time_to_ms(le['start'])}_{time_to_ms(le['end'])}"
        cn_lyrics_map.setdefault(key, []).append(le)

    all_lyric_keys = set(jp_lyrics_map) | set(cn_lyrics_map)
    for key in sorted(all_lyric_keys, key=lambda k: int(k.split('_')[0])):
        jp_entry = jp_lyrics_map.get(key, [None])[0]
        cn_entry = cn_lyrics_map.get(key, [None])[0]
        jp_text = jp_entry['text_clean'] if jp_entry else ''
        cn_text = cn_entry['text_clean'] if cn_entry else ''
        start = (jp_entry or cn_entry)['start']
        end = (jp_entry or cn_entry)['end']
        if jp_text or cn_text:
            pairs.append((start, end, jp_text, cn_text))

    # 5d. Annotation entries
    for ae in annotations:
        style_label = f"##注释 [{ae['style']}]"
        pairs.append((ae['start'], ae['end'], '', f"{style_label}{ae['text_clean']}"))

    for oe in other:
        pairs.append((oe['start'], oe['end'], oe['text_clean'], oe['text_clean']))

    # Sort by start time, filter empty pairs
    pairs.sort(key=lambda e: time_to_ms(e[0]))

    orig_entries: list[tuple[int, str, str, str]] = []
    trans_entries: list[tuple[int, str, str, str]] = []
    for i, (start, end, ot, tt) in enumerate(pairs, 1):
        if not ot.strip() and not tt.strip():
            continue
        orig_entries.append((i, start, end, ot))
        trans_entries.append((i, start, end, tt))

    # 6. Write SRT files
    output_srt(orig_entries, str(out_orig))
    output_srt(trans_entries, str(out_trans))

    print(f"\n{'='*60}")
    print(f"完成! 共 {len(orig_entries)} 条目")
    print(f"  原文: {out_orig}")
    print(f"  译文: {out_trans}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
