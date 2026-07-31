"""Validate the final bilingual ASS"""

import re

ASS_PATH = r"\\Eden_ds\sex\欧美\视频\#Show\V娘的故事\V娘的故事.V.-.The.Hot.One.(1978).1080p.AC3.h264.bilingual.ass"

with open(ASS_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.splitlines()

styles = [l for l in lines if l.startswith('Style:')]
print(f"样式数量: {len(styles)}")
for s in styles:
    print(f"  {s.split(',')[0]}")

dialogues = [l for l in lines if l.startswith('Dialogue:')]
print(f"\n对话条目数: {len(dialogues)}")

style_names = set(s.split(',')[0].replace('Style: ', '') for s in styles)
print(f"样式集合: {style_names}")

has_roriginal = 0
for d in dialogues:
    parts = d.split(',')
    style = parts[3]
    if '\\rOriginal' in d and style != '注释':
        has_roriginal += 1

print(f"正确使用 \\rOriginal 的对话: {has_roriginal}")

print(f"\n=== 长条目检查 ===")
long_cn = 0
long_en = 0
for d in dialogues:
    text = d.split(',,')[-1] if ',,' in d else ''
    text = re.sub(r'\{[^}]*\}', '', text)
    text = text.replace('\\N', ' ').replace('\\rOriginal', '')
    cn_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    en_chars = len(re.sub(r'[\u4e00-\u9fff]', '', text))
    if cn_chars > 40:
        long_cn += 1
    if en_chars > 70:
        long_en += 1

print(f"超过 40 中文字符: {long_cn}")
print(f"超过 70 英文字符: {long_en}")

print(f"\n=== 注释样式条目 ===")
anno = 0
for d in dialogues:
    parts = d.split(',')
    if parts[3] == '注释':
        anno += 1
print(f"注释样式条目数: {anno}")

ok = long_cn == 0 and long_en == 0
print(f"\n=== 总体: {'全部通过' if ok else '需要修复'} ===")
