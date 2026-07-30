from __future__ import annotations

from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "configs"


def _load_toml() -> dict:
    import tomllib

    path = BASE / "config.toml"
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except (FileNotFoundError, tomllib.TOMLDecodeError):
        return {}


def _load_text_dir(rel_dir: str) -> dict:
    result: dict[str, str] = {}
    d = BASE / rel_dir
    if not d.is_dir():
        return result
    for p in sorted(d.iterdir()):
        if p.suffix == ".txt":
            try:
                result[p.stem] = p.read_text(encoding="utf-8")
            except OSError:
                pass
    return result


_config: dict | None = None


def _get_all() -> dict:
    global _config
    if _config is not None:
        return _config

    cfg = _load_toml()
    prompts = _load_text_dir("prompts")
    if prompts:
        cfg["prompts"] = prompts
    ass_cfg = _load_text_dir("ass")
    if ass_cfg:
        cfg["ass"] = ass_cfg
    _config = cfg
    return cfg


def _traverse(*keys: str) -> tuple[object, bool]:
    cfg = _get_all()
    for key in keys:
        if not isinstance(cfg, dict):
            return None, False
        if key not in cfg:
            return None, False
        cfg = cfg[key]
    return cfg, True


def get(*keys: str, fallback: str = "") -> str:
    val, ok = _traverse(*keys)
    if not ok or isinstance(val, dict):
        return fallback
    return str(val)


def get_int(*keys: str, fallback: int = 0) -> int:
    val, ok = _traverse(*keys)
    if not ok or isinstance(val, dict):
        return fallback
    return int(val)


def get_float(*keys: str, fallback: float = 0.0) -> float:
    val, ok = _traverse(*keys)
    if not ok or isinstance(val, dict):
        return fallback
    return float(val)


def get_bool(*keys: str, fallback: bool = False) -> bool:
    val, ok = _traverse(*keys)
    if not ok or isinstance(val, dict):
        return fallback
    return bool(val)


def get_list(*keys: str) -> list:
    val, ok = _traverse(*keys)
    if not ok or not isinstance(val, (list, tuple)):
        return []
    return list(val)


def get_section(*keys: str) -> dict:
    val, ok = _traverse(*keys)
    if not ok or not isinstance(val, dict):
        return {}
    return dict(val)
