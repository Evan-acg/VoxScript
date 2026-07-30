from __future__ import annotations

import shutil
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from config import get
from progress import ProgressCallback, ProgressEvent, null_callback


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass
class MediaInfo:
    duration: float
    path: str


@dataclass
class AudioStream:
    index: int
    codec: str
    language: str | None
    sample_rate: int
    channels: int
    title: str | None


@dataclass
class TranscriptionResult:
    segments: list[Segment] = field(default_factory=list)
    language: str = ""


class AudioExtractor(Protocol):
    @abstractmethod
    def extract(
        self,
        video_path: str,
        output_path: str,
        *,
        stream_index: int | None = None,
        on_progress: ProgressCallback = null_callback,
    ) -> MediaInfo: ...

    @abstractmethod
    def list_audio_streams(self, video_path: str) -> list[AudioStream]: ...


class Transcriber(Protocol):
    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        *,
        model_name: str,
        language: str | None,
        device: str,
        on_progress: ProgressCallback = null_callback,
    ) -> TranscriptionResult: ...


class Formatter(Protocol):
    extension: str

    @abstractmethod
    def format(self, segments: list[Segment]) -> str: ...


@dataclass
class Options:
    model_name: str = "base"
    language: str | None = None
    device: str = "cuda"
    output_dir: str = "./output"
    keep_audio: bool = False
    track_index: int | None = None


class VoxScriptPipeline:
    def __init__(
        self,
        audio_extractor: AudioExtractor,
        transcriber: Transcriber,
        formatter: Formatter,
    ) -> None:
        self._audio_extractor = audio_extractor
        self._transcriber = transcriber
        self._formatter = formatter

    def run(
        self,
        input_path: str,
        opts: Options,
        on_progress: ProgressCallback = null_callback,
    ) -> str:
        import tempfile
        from pathlib import Path

        from audio import is_audio_only

        video_name = Path(input_path).stem
        audio_path = ""

        if is_audio_only(input_path):
            on_progress(
                ProgressEvent("audio_check", 1, 1, "Audio-only file, skipping extraction")
            )
            audio_path = input_path
            result = self._transcriber.transcribe(
                audio_path,
                model_name=opts.model_name,
                language=opts.language,
                device=opts.device,
                on_progress=on_progress,
            )
        else:
            with tempfile.TemporaryDirectory(
                prefix=get("logging", "temp_prefix", fallback="voxscript_")
            ) as tmpdir:
                output_wav = Path(tmpdir, f"{video_name}.wav")
                audio_info = self._audio_extractor.extract(
                    str(input_path), str(output_wav),
                    stream_index=opts.track_index,
                    on_progress=on_progress,
                )

                if opts.keep_audio:
                    keep_path = Path(opts.output_dir, f"{video_name}.wav")
                    keep_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(output_wav), str(keep_path))
                    audio_path = str(keep_path)
                else:
                    audio_path = audio_info.path

                result = self._transcriber.transcribe(
                    audio_path,
                    model_name=opts.model_name,
                    language=opts.language,
                    device=opts.device,
                    on_progress=on_progress,
                )

        subtitle_path = Path(
            opts.output_dir, f"{video_name}.{self._formatter.extension}"
        )
        subtitle_path.parent.mkdir(parents=True, exist_ok=True)
        content = self._formatter.format(result.segments)
        subtitle_path.write_text(content, encoding="utf-8")

        on_progress(
            ProgressEvent(
                "voxscript",
                1,
                1,
                f"Subtitle saved to {subtitle_path}",
            )
        )

        return str(subtitle_path)
