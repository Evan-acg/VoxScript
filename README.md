# VoxScript

使用 FFmpeg、WhisperX 和 LLM 自动修复 ASS 字幕的全自动工具。

程序不会要求人工审核，也不会覆盖输入的 `original.ass`。不确定的项目采用保守策略保留原事件，并写入自动报告。

## 依赖

- Python `>=3.12`
- `uv`
- FFmpeg 和 FFprobe，并且位于 `PATH`
- OpenAI 兼容的 LLM API
- CUDA 环境可选，CPU 模式也可以运行

安装 Python 依赖：

```bash
uv sync
```

设置 API key：

```powershell
$env:OPENAI_API_KEY = "your-api-key"
```

也可以使用项目原有的 `VOX_API_KEY`。

## 使用

```bash
uv run starter.py repair --video video.mkv --subtitle original.ass --output repaired.ass --chunk-minutes 10
```

常用选项：

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--video` | 必填 | 视频或音频文件 |
| `--subtitle` | 必填 | 原始 ASS 文件 |
| `--output` | `<video>.repaired.ass` | 最终 ASS 文件 |
| `--chunk-minutes` | `10` | 每个主体分段的分钟数 |
| `--context-seconds` | `10` | 分段前后上下文秒数 |
| `--model` | `base` | WhisperX 模型 |
| `--language` | 自动检测 | ASR 源语言 |
| `--target-language` | `auto` | 现有字幕的目标语言 |
| `--device` | 配置值 | `cpu` 或 `cuda` |
| `--vad-method` | `silero` | VAD 后端，`silero` 更快 |
| `--batch-size` | `16` | WhisperX 批量大小 |
| `--track` | 第一个音轨 | FFmpeg 音频流索引 |
| `--llm-model` | 配置值 | LLM 模型名 |
| `--keep-artifacts` | 关闭 | 保留中间文件 |
| `--work-dir` | 隐藏临时目录 | 中间文件目录 |

没有指定 `--track` 时自动使用第一个音频流，不进行交互式选择。

## 处理流程

```text
video.mkv + original.ass
        ↓
固定分段和上下文
        ↓
FFmpeg 提取分段音频
        ↓
WhisperX 生成 ASR 时间证据
        ↓
LLM 返回字幕 ID 和 ASR ID 的修改操作
        ↓
程序计算时间并回写 ASS
        ↓
repaired.ass
```

LLM 不生成时间戳。程序使用第一个 ASR 片段的开始时间和最后一个 ASR 片段的结束时间，并分别添加 `0.10` 秒和 `0.20` 秒缓冲。

ASS 的原始样式、层级、名称、边距、效果和内嵌标签会被保留。程序只修改事件的开始时间、结束时间和文本。

## 中间结果

使用 `--keep-artifacts` 时会保留：

```text
asr.json       # ASR 片段和绝对时间
review.json    # 自动应用、保留和错误报告
preview.ass    # 写入最终文件前的 ASS
```

默认情况下，完全成功且没有保守降级时删除这些中间文件，只保留 `repaired.ass`。如果发生分段错误或存在 `review`/`delete` 等未自动应用项目，程序会保留报告并在命令输出中显示路径。

## 自动降级规则

- `keep` 保留文本，只更新时间。
- `revise` 修改文本并更新时间。
- `insert` 自动新增明确缺失的台词。
- `delete` 保留原字幕，不自动删除。
- `review` 保留原字幕的原文本和时间。
- `split`、`merge` 和标签校验失败的修改会被保留为原事件。
- 分段或 LLM 失败时，该分段保持原字幕，其他分段继续处理。
- 输入损坏、无法读取 ASS 或无法加载模型等致命错误会使命令失败。

## 测试

```bash
uv run pytest
uv run starter.py --help
uv run starter.py repair --help
```
