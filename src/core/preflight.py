from __future__ import annotations

import hashlib
import shutil

from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache
from pydantic import BaseModel

from src.core.config import PROJECT_ROOT, AppConfig

MARKER_PATH = PROJECT_ROOT / ".preflight.ok"


class PreflightResult(BaseModel):
    name: str
    ok: bool
    detail: str


def _fingerprint(config: AppConfig) -> str:
    payload = f"{config.model_dir}\0{config.model_name}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _marker_matches(config: AppConfig) -> bool:
    if not MARKER_PATH.is_file():
        return False
    try:
        return MARKER_PATH.read_text(encoding="utf-8").strip() == _fingerprint(config)
    except OSError:
        return False


def _write_marker(config: AppConfig) -> None:
    MARKER_PATH.write_text(_fingerprint(config), encoding="utf-8")


def run_preflight_if_needed(
    config: AppConfig,
    force: bool = False,
) -> list[PreflightResult] | None:
    if not force and _marker_matches(config):
        return None

    results = run_preflight(config)
    if all(result.ok for result in results):
        _write_marker(config)
    return results


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
