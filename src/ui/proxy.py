from __future__ import annotations

import io
from typing import IO

from rich.ansi import AnsiDecoder
from rich.console import Console
from rich.text import Text


class ConsoleProxy(io.TextIOBase):
    """Routes third-party stdout/stderr writes into the console, decoding ANSI
    escapes into rich styles (colors survive on VT and legacy consoles)."""

    def __init__(self, console: Console, file: IO[str]) -> None:
        self.__console = console
        self.__file = file
        self.__buffer: list[str] = []
        self.__ansi_decoder = AnsiDecoder()

    def __getattr__(self, name: str):
        return getattr(self.__file, name)

    @property
    def rich_proxied_file(self) -> IO[str]:
        return self.__file

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError(f"write() argument must be str, not {type(text).__name__}")
        buffer = self.__buffer
        lines: list[str] = []
        while text:
            line, new_line, text = text.partition("\n")
            if new_line:
                lines.append("".join(buffer) + line)
                buffer.clear()
            else:
                buffer.append(line)
                break
        if lines:
            with self.__console:
                output = Text("\n").join(
                    self.__ansi_decoder.decode_line(line) for line in lines
                )
                self.__console.print(output)
        return len(text)

    def flush(self) -> None:
        output = "".join(self.__buffer)
        if output:
            with self.__console:
                self.__console.print(self.__ansi_decoder.decode_line(output))
        del self.__buffer[:]

    def fileno(self) -> int:
        return self.__file.fileno()

    def isatty(self) -> bool:
        return self.__file.isatty()
