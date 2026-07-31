from __future__ import annotations

import shutil

from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache
from pydantic import BaseModel

from src.core.config import AppConfig


class PreflightResult(BaseModel):
    name: str
    ok: bool
    detail: str


def run_preflight(config: AppConfig) -> list[PreflightResult]:
    results: list[PreflightResult] = []

    ffmpeg = shutil.which("ffmpeg")
    results.append(
        PreflightResult(
            name="ffmpeg",
            ok=ffmpeg is not None,
            detail=ffmpeg or "ffmpeg not found on PATH",
        )
    )

    ffprobe = shutil.which("ffprobe")
    results.append(
        PreflightResult(
            name="ffprobe",
            ok=ffprobe is not None,
            detail=ffprobe or "ffprobe not found on PATH",
        )
    )

    if not config.model_dir.is_dir():
        results.append(
            PreflightResult(
                name="whisperx model",
                ok=False,
                detail=f"model directory does not exist: {config.model_dir}",
            )
        )
    else:
        repo_id = f"Systran/faster-whisper-{config.model_name}"
        resolved = try_to_load_from_cache(
            repo_id,
            "model.bin",
            cache_dir=str(config.model_dir),
        )
        ok = resolved is not None and resolved is not _CACHED_NO_EXIST
        results.append(
            PreflightResult(
                name="whisperx model",
                ok=ok,
                detail=(
                    f"{repo_id} resolved in cache"
                    if ok
                    else f"{repo_id} not found in {config.model_dir}"
                ),
            )
        )

    return results
