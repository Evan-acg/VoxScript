from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from rich.console import Console
from rich.prompt import IntPrompt
from rich.table import Table

from src.entity.context import PipelineContext
from src.entity.proof import ProofArgs

console = Console()


class AudioHandler:
    def __init__(self, args: ProofArgs, context: PipelineContext) -> None:
        self.args = args
        self.context = context

    def extract_audio(self) -> PipelineContext:
        streams = self._probe_audio_streams()
        stream_index = self._select_audio_stream(streams)

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found on PATH")

        work_dir = Path(tempfile.mkdtemp(prefix="voxscript_"))
        output_path = work_dir / f"{self.args.input_path.stem}.wav"

        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(self.args.input_path),
            "-map",
            f"0:{stream_index}",
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip()
            raise RuntimeError(f"ffmpeg failed: {detail or 'unknown error'}")

        self.context.audio_path = output_path
        self.context.audio_track = stream_index
        return self.context

    def _probe_audio_streams(self) -> list[dict]:
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            raise RuntimeError("ffprobe not found on PATH")

        cmd = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index,codec_name,channels:stream_tags=language",
            "-of",
            "json",
            str(self.args.input_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip()
            raise RuntimeError(f"ffprobe failed: {detail or 'unknown error'}")
        data = json.loads(proc.stdout or "{}")
        return data.get("streams", [])

    def _select_audio_stream(self, streams: list[dict]) -> int:
        if not streams:
            raise RuntimeError("no audio track found in input file")

        if len(streams) == 1:
            index = streams[0]["index"]
            console.print(f"Audio track: 1 track, auto-selected index {index}")
            return index

        console.print("Multiple audio tracks found:")
        table = Table(title="Audio tracks")
        table.add_column("#", justify="right")
        table.add_column("Index")
        table.add_column("Codec")
        table.add_column("Channels")
        table.add_column("Language")
        for ordinal, stream in enumerate(streams, start=1):
            table.add_row(
                str(ordinal),
                str(stream.get("index")),
                str(stream.get("codec_name", "-")),
                str(stream.get("channels", "-")),
                str(stream.get("tags", {}).get("language", "-")),
            )
        console.print(table)

        while True:
            choice = IntPrompt.ask(
                f"Select audio track [1-{len(streams)}]",
                default=1,
                show_default=False,
            )
            if 1 <= choice <= len(streams):
                return streams[choice - 1]["index"]
            console.print(
                f"[red]Invalid selection: {choice}. Expected 1-{len(streams)}.[/red]"
            )
