from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, field_validator, model_validator

_ALLOWED_SUBTITLE_FORMATS = ("srt", "ass", "ssa")


class ProofArgs(BaseModel):
    input_path: Path
    subtitle_paths: list[Path]
    output_path: Path | None = None
    model_dir: Path

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
        self.output_path = self.output_path or self.input_path.with_suffix(".ass")
        return self
