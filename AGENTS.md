# AGENTS.md

## 命令与入口

- 入口文件：`starter.py`（仓库根目录，click CLI）。它自行把仓库根目录插入 `sys.path`，包从未被安装——必须从仓库根目录 `uv run starter.py -i <视频> -s <字幕>` 运行，不能用 `python -m src...`。
- `midterm.bat` 只是开发期手动验证脚本，不是入口：顶部硬编码 `INPUT`/`SUBTITLE` 路径，内部设置 `HF_ENDPOINT=https://hf-mirror.com`，失败 `pause`。
- 项目无 pytest/ruff/mypy 等任何测试或 lint 配置——验证方式只有手动运行；不要提议加测试框架（用户已明确不要测试）。

## 架构

- 分层：`src/command`（编排）→ `src/handler`（asr/audio/subtitle）→ `src/entity`（pydantic 模型）→ `src/parser`（字幕解析策略）→ `src/ui`（rich Live 面板）；`src/core` 为 events/config/preflight。
- 流水线步骤（`cli.py::_run_pipeline`）：preflight → extract_audio → load_model → transcribe → normalize_subtitles（generate_ass 仅占位，未实现）。
- `EventBus` 解耦流水线与 UI（step_started/completed/failed + log + progress）。
- 坑：每个发送 step 事件的步骤必须同时存在于 `src/ui/steps.py` 的 `PIPELINE_STEPS`，否则 `StepsView` KeyError（normalize_subtitles 曾触发）。
- `PipelineContext` 是单例可变对象，按引用贯穿所有 handler；`AsrHandler` 须在 load_model 步骤构造一次，transcribe 步骤复用同一实例。中间产物（.wav/.srt）目录由 `context.run_dir` 统一管理：`vox_YYYYMMDD_HHMMSS` 子目录，根目录取 `config.yaml` 的 `work_dir`，未配置回退系统 temp。

## 约定

- 错误链：handler 抛 `RuntimeError` → cli.py 捕获 → `bus.step_failed(...)` → 转 `click.ClickException`。
- 字幕仅支持 .srt/.ass/.ssa（未知扩展名抛 ValueError）；读取按 utf-8-sig → gb18030 回退；写出统一 UTF-8。
- whisperx：CPU + int8 + 30s 分块；`_load_model` 优先本地缓存（`local_files_only=True`），`LocalEntryNotFoundError` 才联网下载；`HF_ENDPOINT` 全局未设置（仅 midterm.bat 内有），错误提示中引用镜像地址。
- 模型配置在 `configs/config.yaml`（model_dir / model_name）；`.preflight.ok` 指纹缓存（model_dir+model_name），改配置会自动重检，`--force-check` 强制重检。
- 代码风格：`from __future__ import annotations`、全程类型标注、除非要求否则不加注释。
