# VoxScript

Extract audio tracks from video and generate subtitles using WhisperX.

> **Requires** `ffmpeg` and `ffprobe` installed on the system and available in PATH.

## Installation

```bash
git clone <repo>
cd VoxScript
uv sync
```

## Usage

```bash
# Basic usage (base model, CUDA, auto-detect language)
voxscript video.mp4

# Specify model and language
voxscript video.mp4 --model small --language zh

# Use CPU and keep extracted audio
voxscript video.mp4 --device cpu --keep-audio

# Custom output directory
voxscript video.mp4 --output-dir ./subtitles
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model` | `base` | Model size: `tiny`, `base`, `small`, `medium`, `large` |
| `-l, --language` | auto | Language code (e.g. `zh`, `en`, `ja`) |
| `-d, --device` | `cuda` | Device: `cpu` or `cuda` |
| `-o, --output-dir` | `.` | Output directory for subtitle file |
| `--keep-audio` | off | Retain the intermediate 16kHz WAV file |

## How It Works

```
Video → [ffmpeg → 16kHz mono WAV] → [WhisperX transcribe + align] → SRT subtitle
```

1. **Audio Extraction**: Uses `ffprobe` for duration, then `ffmpeg` extracts PCM 16kHz mono WAV with real-time progress tracking.
2. **Speech Recognition**: WhisperX transcribes audio with VAD-based segmentation and word-level timestamp alignment.
3. **Subtitle Export**: Writes SRT format to disk.

## Dependencies

- **Python**: `>=3.12`
- **PyPI**: `click`, `rich`, `whisperx`
- **System**: `ffmpeg`, `ffprobe`

## Architecture

| Module | Pattern | Responsibility |
|--------|---------|---------------|
| `pipeline.py` | Facade | Orchestrates the 3-stage workflow |
| `audio.py` | — | FFmpeg audio extraction with progress |
| `subtitle.py` | Strategy | WhisperX transcription + SRT formatting |
| `progress.py` | Observer/Callback | Rich-based progress bar, decoupled via events |
| `cli.py` | — | Click CLI, dependency injection assembly |

All cross-module interfaces (`AudioExtractor`, `Transcriber`, `Formatter`) are defined as Protocols in `pipeline.py`, following Dependency Inversion Principle.
