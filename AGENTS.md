# VoxScript - Agent Guide

**使用中文回复**

本项目使用uv进行包管理， 禁止直接使用python运行脚本。

## Quick Start

```bash
uv sync
uv run starter.py --help
```

## Commands

- `uv sync` — install dependencies
- `uv run starter.py repair --video <file> --subtitle <file> --output repaired.ass` — automatically repair ASS
- `uv add <package>` — add dependency
- `uv run python -c "..."` — run inline script
- `uv tree` — view dependency tree

## Project Structure

```
VoxScript/
├── starter.py         # Entry point
├── src/
│   ├── cli.py         # Click CLI + dependency assembly
│   ├── repair/        # ASS, chunks, ASR, LLM and workflow
│   ├── commands/      # CLI commands
│   └── progress/      # Rich progress reporter
├── pyproject.toml
├── AGENTS.md
└── README.md
```

## Conventions

- `from __future__ import annotations` in all files
- Type hints everywhere
- dataclass for models (frozen when immutable)
- Protocol for DI interfaces
- Custom exceptions per module (inherit `Exception`)
- `ProgressCallback` for cross-module progress, not direct `rich` dependency
- Single CLI entry in `cli.py`, assembly of deps only
- No comments in code (unless required)

## Tests

- Tests directory: `tests/`
- Run: `uv run pytest`

## Key Dependencies

- `click` — CLI framework
- `rich` — progress bar
- `whisperx` — speech-to-text (includes torch, faster-whisper)
- `ffmpeg` / `ffprobe` — system dependency for audio extraction
- `pysubs2` — ASS parsing and writing
