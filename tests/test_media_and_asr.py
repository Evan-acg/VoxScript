from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from src.repair.asr import WhisperXASR
from src.repair.media import build_extract_command


def test_extract_command_uses_relative_duration_for_each_chunk() -> None:
    command = build_extract_command(
        "ffmpeg",
        "video.mkv",
        "chunk.wav",
        stream_index=2,
        start=10.0,
        end=20.0,
        sample_rate=16000,
        channels=1,
    )

    assert command[0] == "ffmpeg"
    assert command[command.index("-map") + 1] == "0:2"
    assert command[command.index("-ss") + 1] == "10.0"
    assert command[command.index("-t") + 1] == "10.0"
    assert command[-1] == "chunk.wav"


def test_whisperx_asr_returns_absolute_segment_times(monkeypatch: object, tmp_path: Path) -> None:
    load_kwargs: dict[str, object] = {}
    transcribe_kwargs: dict[str, object] = {}
    align_calls = 0

    class FakeAudio:
        shape = (80000,)

    class FakeModel:
        def transcribe(self, audio: object, **kwargs: object) -> dict[str, object]:
            transcribe_kwargs.update(kwargs)
            return {
                "language": "en",
                "segments": [
                    {"start": 1.0, "end": 2.0, "text": "hello"},
                ],
            }

    def load_model(*args: object, **kwargs: object) -> FakeModel:
        load_kwargs.update(kwargs)
        return FakeModel()

    def load_align_model(*args: object, **kwargs: object) -> tuple[None, None]:
        nonlocal align_calls
        align_calls += 1
        return None, None

    fake_whisperx = SimpleNamespace(
        load_audio=lambda path: FakeAudio(),
        load_model=load_model,
        load_align_model=load_align_model,
    )
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)

    audio_path = tmp_path / "chunk.wav"
    audio_path.write_bytes(b"audio")
    result = WhisperXASR().transcribe(
        str(audio_path),
        offset=600.0,
        chunk_id=3,
        model_name="tiny",
        language="en",
        device="cpu",
        batch_size=16,
        align=True,
    )

    assert result == [
        result[0].__class__(
            id=0,
            start=601.0,
            end=602.0,
            text="hello",
            chunk_id=3,
        )
    ]
    assert load_kwargs["vad_method"] == "silero"
    assert transcribe_kwargs["batch_size"] == 16
    assert align_calls == 1


def test_whisperx_asr_falls_back_to_raw_segments_without_alignment(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    class FakeAudio:
        shape = (80000,)

    class FakeModel:
        def transcribe(self, audio: object, **kwargs: object) -> dict[str, object]:
            return {
                "language": "xx",
                "segments": [{"start": 0.5, "end": 1.0, "text": "fallback"}],
            }

    def fail_alignment(*args: object, **kwargs: object) -> object:
        raise RuntimeError("language has no alignment model")

    fake_whisperx = SimpleNamespace(
        load_audio=lambda path: FakeAudio(),
        load_model=lambda *args, **kwargs: FakeModel(),
        load_align_model=fail_alignment,
    )
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)

    audio_path = tmp_path / "chunk.wav"
    audio_path.write_bytes(b"audio")
    result = WhisperXASR().transcribe(
        str(audio_path),
        offset=10.0,
        chunk_id=1,
        model_name="tiny",
        language="xx",
        device="cpu",
        align=True,
    )

    assert result[0].text == "fallback"
    assert result[0].start == 10.5
