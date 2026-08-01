from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, field_validator, model_validator

_ALLOWED_SUBTITLE_FORMATS = ("srt", "ass", "ssa")


class ProofArgs(BaseModel):
    input_path: Path
    subtitle_paths: list[Path]
    output_path: Path | None = None
    output_is_dir: bool = False
    model_dir: Path
    api_key: str | None = None
    api_key_env: str | None = None

    @field_validator("input_path")
    @classmethod
    def _validate_input(cls, value: Path) -> Path:
        if not value.is_file():
            raise ValueError(f"input file does not exist: {value}")
        return value

    @field_validator("model_dir")
    @classmethod
    def _validate_model_dir(cls, value: Path) -> Path:
        if not value.is_dir():
            raise ValueError(f"model directory does not exist: {value}")
        return value

    @field_validator("subtitle_paths")
    @classmethod
    def _validate_subtitles(cls, value: list[Path]) -> list[Path]:
        for path in value:
            if path.suffix.lstrip(".") not in _ALLOWED_SUBTITLE_FORMATS:
                raise ValueError(
                    f"unsupported subtitle format: {path.suffix or '<none>'}; "
                    f"expected one of: {', '.join(_ALLOWED_SUBTITLE_FORMATS)}"
                )
        return value

    @model_validator(mode="after")
    def _resolve_output(self) -> ProofArgs:
        output = self.output_path or self.input_path.with_suffix(".ass")
        if output.is_dir():
            output = output / f"{self.input_path.stem}.Repaired.ass"
        elif self.output_is_dir:
            output.mkdir(parents=True, exist_ok=True)
            output = output / f"{self.input_path.stem}.ass"
        else:
            if output.suffix.lower() != ".ass":
                raise ValueError(f"output path must end with .ass: {output}")
            parent = output.parent
            if not parent.exists() or not parent.is_dir():
                raise ValueError(f"output directory does not exist: {parent}")
        if output.is_dir():
            raise ValueError(f"output path is a directory: {output}")
        if output == self.input_path:
            raise ValueError("output path must differ from the input file")
        self.output_path = output
        return self

    @model_validator(mode="after")
    def _resolve_api_key(self) -> ProofArgs:
        if self.api_key_env is None:
            return self
        if self.api_key:
            return self
        resolved = os.environ.get(self.api_key_env, "")
        if not resolved:
            raise ValueError(
                f"no LLM api key: pass --api-key or set the "
                f"{self.api_key_env} environment variable"
            )
        self.api_key = resolved
        return self
