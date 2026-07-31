"""Validate final bilingual ASS - detailed check"""

import re

PATH = r"\\Eden_ds\sex\欧美\视频\#Show\V娘的故事\V娘的故事.V.-.The.Hot.One.(1978).1080p.AC3.h264.bilingual.ass"

with open(PATH, 'r', encoding='utf-8') as f:
    content = f.read()

dialogues = [l for l in content.splitlines() if l.startswith('Dialogue:')]
print(f'Total entries: {len(dialogues)}')

max_en = 0
max_cn = 0
all_ok = True

for i, d in enumerate(dialogues, 1):
    m = re.match(
        r'Dialogue:\s*(\d+),([^,]+),([^,]+),([^,]*),([^,]*),(\d+),(\d+),(\d+),([^,]*),(.+)', d
    )
    if not m:
        continue
    text = m.group(10)
    style = m.group(4)
    text_clean = re.sub(r'\{[^}]*\}', '', text)

    if style == '\u6ce8\u91ca':  # '注释'
        if len(text_clean.replace(r'\N', ' ')) > 40:
            print(f'WARN Annotation long: {text_clean}')
        continue

    parts = text_clean.split(r'\N')
    if len(parts) >= 2:
        cn_part = re.sub(r'\\rOriginal\s*', '', parts[0]).strip()
        en_part = re.sub(r'\\rOriginal\s*', '', parts[1]).strip()
    else:
        cn_part = parts[0]
        en_part = ''

    en_len = len(en_part)
    cn_len = len(cn_part)
    max_en = max(max_en, en_len)
    max_cn = max(max_cn, cn_len)

    if en_len > 70:
        print(f'EN too long ({en_len} chars) at entry {i}: {en_part[:80]}')
        all_ok = False
    if cn_len > 40:
        print(f'CN too long ({cn_len} chars) at entry {i}: {cn_part[:60]}')
        all_ok = False

print(f'Max EN length: {max_en}')
print(f'Max CN length: {max_cn}')
print(f'All ok: {all_ok}')

# Show annotation entries
anno_count = 0
for d in dialogues:
    if d.split(',')[3] == '\u6ce8\u91ca':
        anno_count += 1
print(f'Annotation entries: {anno_count}')
