from __future__ import annotations

from typing import Any, Callable

from rich.console import Console
from rich.prompt import IntPrompt
from rich.table import Table


class TrackPrompt:
    def __init__(self, console: Console) -> None:
        self._console = console

    def ask(
        self,
        streams: list[dict[str, Any]],
        pause: Callable[[], None],
        resume: Callable[[], None],
    ) -> int:
        pause()
        self._console.print("Multiple audio tracks found:")
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
        self._console.print(table)
        count = len(streams)
        while True:
            choice = IntPrompt.ask(
                f"Select audio track [1-{count}]",
                default=1,
                show_default=False,
                console=self._console,
            )
            if 1 <= choice <= count:
                break
            self._console.print(
                f"[red]Invalid selection: {choice}. Expected 1-{count}.[/red]"
            )
        resume()
        return streams[choice - 1]["index"]
