# VoxScript 自动修复 MVP

当前流程：

```text
视频或音频 + original.ass
→ 固定分段
→ FFmpeg 提取音频
→ WhisperX 生成 ASR 时间证据
→ LLM 返回匹配和校对操作
→ 程序自动回写 ASS
→ repaired.ass
```

不包含人工审核、说话人分离、OCR、质量评分、镜头检测或复杂漂移建模。

不确定项目使用保守策略保留原字幕，并写入 `review.json`。
