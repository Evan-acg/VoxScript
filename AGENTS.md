# AGENTS.md

## 命令与入口

- 入口文件：`starter.py`（仓库根目录，click CLI）。它自行把仓库根目录插入 `sys.path`，包从未被安装——必须从仓库根目录 `uv run starter.py -i <视频> -s <字幕>` 运行，不能用 `python -m src...`。
- `midterm.bat` 只是开发期手动验证脚本，不是入口：顶部硬编码 `INPUT`/`SUBTITLE` 路径，内部设置 `HF_ENDPOINT=https://hf-mirror.com`，失败 `pause`。
- 项目无 pytest/ruff/mypy 等任何测试或 lint 配置——验证方式只有手动运行；不要提议加测试框架（用户已明确不要测试）。

## 架构

- 分层：`src/command`（编排：`cli.py` 入口 + `pipeline.py` 管线）→ `src/handler`（步骤适配：事件/上下文，无实际处理逻辑）→ `src/service`（纯功能层：ffmpeg/whisperx/字幕解析，不依赖 bus/context/args，可独立调用）→ `src/entity`（pydantic 模型）→ `src/parser`（字幕解析策略）→ `src/ui`（rich Live 面板）；`src/core` 为 events/config/preflight。
- 管线模式：`src/command/pipeline.py` 的 `build_pipeline()` 是步骤清单的唯一事实来源（preflight → extract_audio → load_model → transcribe → normalize_subtitles；generate_ass 未实现，实现时再挂步骤）。`Pipeline` 统一管理步骤生命周期：`step_started` → `run()` →（`RuntimeError`/`ClickException` → `step_failed`）→ `step_completed`，handler 全部在其中构造，`AsrHandler` 一次构造、load_model/transcribe 两步复用。`cli.py` 仅负责：构建 args/config/bus/dashboard → `dashboard.set_steps(pipeline.names)` 同步 UI 步骤清单 → `pipeline.run()` → 打印汇总（读 `pipeline.context`）。
- `EventBus` 解耦流水线与 UI（step_started/completed/failed + log + progress）。
- Dashboard 不启用 rich `Live` 自带的 IO 重定向（其 `FileProxy.flush` 会把无换行文本按 markup 解析，拆散 ANSI 转义产生裸 `[0m`）；改用 `src/ui/proxy.py` 的 `ConsoleProxy`（`markup=False` 原样输出，`rich_proxied_file` 契约防递归），`Dashboard.start()/stop()` 幂等安装/恢复 stdout/stderr，第三方输出（whisperx print、tqdm 彩色条）照常显示且字节流干净。
- 坑：步骤清单只存在于 `build_pipeline` 一处（不再有 `ui/steps.py` 的 `PIPELINE_STEPS` 常量）；`Dashboard.set_steps()` 必须在 `pipeline.run()` 之前调用，否则 `StepsView` KeyError。
- `PipelineContext` 是单例可变对象，按引用贯穿所有 handler。中间产物（.wav/.srt）目录由 `context.run_dir` 统一管理：`vox_YYYYMMDD_HHMMSS` 子目录，根目录取 `config.yaml` 的 `work_dir`，未配置回退系统 temp。

## 约定

- 错误链：handler 抛 `RuntimeError` → `Pipeline.run()` 捕获 → `bus.step_failed(...)` → 转 `click.ClickException`（preflight 直抛 ClickException，`Pipeline` 补发 `step_failed` 后透传）。
- 功能与编排分离：实际处理逻辑全部在 `src/service`，handler 只做事件/上下文适配；service 函数接收普通参数（Path、可选 `on_progress`/`on_log` 回调），可脱离 handler 独立调用。
- 字幕仅支持 .srt/.ass/.ssa（未知扩展名抛 ValueError）；读取按 utf-8-sig → gb18030 回退；写出统一 UTF-8。
- whisperx：CPU + int8 + 30s 分块；`_load_model` 优先本地缓存（`local_files_only=True`），`LocalEntryNotFoundError` 才联网下载；`HF_ENDPOINT` 全局未设置（仅 midterm.bat 内有），错误提示中引用镜像地址。
- 模型配置在 `configs/config.yaml`（model_dir / model_name）；`.preflight.ok` 指纹缓存（model_dir+model_name），改配置会自动重检，`--force-check` 强制重检。
- 代码风格：`from __future__ import annotations`、全程类型标注、除非要求否则不加注释。
