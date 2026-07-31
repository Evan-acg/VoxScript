from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "config.yaml"


class AppConfig(BaseModel):
    model_dir: Path

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)
