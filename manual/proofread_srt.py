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
    # ── 字幕组广告 → 加 ##注释 前缀 ──
    1: "##注释下载自 YTS.MX",
    2: "##注释YIFY 电影官方网站：YTS.MX",

    # ── 语义修正 ──
    3: "难以置信。\n就在数学课上。",

    16: "她的隆胸手术",

    25: "我父母正常得\n让人沮丧。",
    26: "真是压抑。",

    30: "找出那个从未做过爱的人——\n一次都没有。",

    46: "像在泳池里打球一样吗？",

    68: "罗曼被当场抓住……",

    84: "我不是以律师身份在说话，伯特兰夫人，",

    109: "他们很敢玩。\n他们怂恿我做的。",

    114: "好吧，为了好玩，\n你们都在拍自己",

    136: "那不算是真正的敢\n除非你拍下来。",

    165: "除了艾滋病、\n避孕……",

    169: "别惊慌，小克莱尔。",

    171: "告诉我我们是什么样的形象\n才让罗曼",

    172: "拍下自己\n在生物课上自慰？",

    184: "“证明一下。",

    185: "我被抓了，\n应该得到加分才对。”",

    186: "“晚安！”",

    188: "“清空近期历史。”",
    189: "“现在清空了。”",

    270: "我想要性，又不想\n扮演体贴的伴侣",

    271: "被迫和同龄女人\n没话找话聊。",

    294: "“我看过了。",
    295: "你说得对。\n你值得更多。”",

    297: "“和我想的一样。”",

    300: "骗他让我很不安。",

    301: "但说实话\n也让我不安。",

    332: "“保持冷静，兄弟。",
    333: "自慰是正常的。”",
    334: "我回了条：\n“你才自慰呢？”",

    341: "“今天想你了。”",

    419: "你在出轨妈妈吗？",

    421: "为什么你要买\n那么多避孕套？",

    430: "就是\n及时抽出来。",

    433: "皮埃尔总说\n我是个意外。",

    442: "我想吃点荔枝。",
    456: "你在外面有人吗？",
    457: "你赌我就是。",

    467: "那就躺在地板上，脱光。",

    472: "他是我的男人玩具。",

    477: "我去谷歌搜一下“反全球化者”。\n这应该能让我冷静。",

    486: "所有指控都撤销了。",

    488: "多亏了科拉莉\n全班都在自拍。",

    489: "他们总不能\n把所有人都停学吧。",

    490: "皮埃尔说\n这是所打飞机学校。",

    491: "还有打飞机妹，我补充道。",

    492: "玛丽的新男友\n她的性满足者，",

    497: "以前性事是禁忌\n是不是反而更好？",

    503: "没关系。",

    511: "靠，我忘带套了。",

    524: "你把一切都拍下来了，\n科拉莉。",

    543: "我是双性恋。",

    548: "关于她的性生活，\n妈妈宣布我们全都无罪。",

    558: "你知道真正的扫兴\n是什么吗？",

    561: "别激动，亲爱的小猫。\n只是个想法而已。",

    562: "别那么叫我。\n太粗俗了。",

    577: "我会把它做完\n来纪念他。",

    585: "我意识到生命短暂，\n所以选了它当职业。",

    592: "不了。\n我九点还有个客人。",

    594: "她九点有个客人，\n她说。",

    597: "意思是我也能\n十一点左右约到人。",

    607: "那你就自己解决呗。",

    621: "我一般在\n浴室里做。",

    628: "在我脑海里，\n你永远那么美。",

    634: "我的手指在四周爱抚。",

    636: "当你全部的注意力\n都集中在那里……",

    639: "前液沾湿了我的下面。",

    644: "想不粗俗\n还真不容易。",

    646: "在我脑海里，和你\n从来就没有粗俗这回事，亲爱的。",

    648: "那些女孩\n总被视为受害者，",

    649: "但她真的很自信。",

    652: "如果那是份理想工作，\n我们早就去干了。",

    653: "和所有事情一样。\n不能一概而论。",

    659: "她全身都整过了。",

    661: "你想让我\n也去隆胸吗？",

    663: "就稍微提一点。",

    667: "现在男人也做整容手术了。",

    668: "你可以轻松\n去掉这个。",

    671: "你会介意\n一起变老吗？",

    672: "它又阻止不了我们变老。",

    673: "这是最后\n我们老得最少的地方。",

    676: "让我兴奋的是\n她也兴奋了。",

    682: "她还说，看得越多，\n发现得越多。",

    683: "你展现得越多，\n就越了解自己。",

    685: "射在外面？",

    687: "没有摄像机的时候，\n我都射在里面。",

    692: "再来一张？",

    693: "再来张全家福。",

    694: "这个家越来越大了。",

    695: "那是马克西姆，花店老板。",

    698: "到头来，\n一切皆有可能。",

    699: "是时候让妈妈打开\n一个新档案了：",

    700: "幸福。",

    701: "在一切搞砸之前。",
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
