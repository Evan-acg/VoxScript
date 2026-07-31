from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pysubs2
from pysubs2.exceptions import FormatAutodetectionError

from .models import SubtitleCue


class AssTagError(ValueError):
    pass


@dataclass
class LoadedSubtitle:
    file: pysubs2.SSAFile
    cues: dict[int, SubtitleCue]
    events_by_id: dict[int, Any]
    cue_id_by_event_id: dict[int, int]
    next_generated_id: int = -1
    encoding: str = "utf-8"

    @property
    def events(self) -> list[Any]:
        return self.file.events

    def add_event(self, event: Any) -> int:
        self.file.events.append(event)
        generated_id = self.next_generated_id
        self.next_generated_id -= 1
        self.events_by_id[generated_id] = event
        self.cue_id_by_event_id[id(event)] = generated_id
        self.cues[generated_id] = SubtitleCue(
            id=generated_id,
            event_index=len(self.file.events) - 1,
            start=event.start / 1000,
            end=event.end / 1000,
            text=event.text,
        )
        return generated_id


_ASS_TOKEN_RE = re.compile(r"(\{[^{}]*\}|\\[Nnh])")
_PLACEHOLDER_RE = re.compile(r"<ASS_TAG_(\d+)>")
_ANY_PLACEHOLDER_RE = re.compile(r"<ASS_TAG_[^>]*>")


def mask_ass_tags(text: str) -> tuple[str, tuple[str, ...]]:
    tokens: list[str] = []

    def replace(match: re.Match[str]) -> str:
        tokens.append(match.group(0))
        return f"<ASS_TAG_{len(tokens) - 1}>"

    return _ASS_TOKEN_RE.sub(replace, text), tuple(tokens)


def restore_ass_tags(text: str, tokens: tuple[str, ...]) -> str:
    placeholders = [int(match.group(1)) for match in _PLACEHOLDER_RE.finditer(text)]
    if placeholders != list(range(len(tokens))):
        raise AssTagError("ASS tag placeholders are missing or reordered")
    for placeholder in _ANY_PLACEHOLDER_RE.findall(text):
        if _PLACEHOLDER_RE.fullmatch(placeholder) is None:
            raise AssTagError(f"unknown ASS tag placeholder: {placeholder}")
    if _ASS_TOKEN_RE.search(text) is not None:
        raise AssTagError("new or unmasked ASS tag found in revised text")

    for index, token in enumerate(tokens):
        placeholder = f"<ASS_TAG_{index}>"
        if text.count(placeholder) != 1:
            raise AssTagError(f"missing or duplicated ASS tag placeholder: {placeholder}")
        text = text.replace(placeholder, token)

    return text


def _load_with_encodings(path: Path) -> tuple[pysubs2.SSAFile, str]:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe"):
        encodings = (
            ("utf-16", "utf-16"),
            ("utf-16-le", "utf-16-le"),
            ("utf-8-sig", "utf-8-sig"),
            ("gb18030", "gb18030"),
        )
    elif raw.startswith(b"\xfe\xff"):
        encodings = (
            ("utf-16", "utf-16-be"),
            ("utf-16-be", "utf-16-be"),
            ("utf-8-sig", "utf-8-sig"),
            ("gb18030", "gb18030"),
        )
    elif b"\x00" in raw[:4096]:
        encodings = (
            ("utf-16-le", "utf-16-le"),
            ("utf-16-be", "utf-16-be"),
            ("utf-8-sig", "utf-8-sig"),
            ("gb18030", "gb18030"),
        )
    else:
        encodings = (
            ("utf-8-sig", "utf-8-sig"),
            ("gb18030", "gb18030"),
            ("utf-16", "utf-16"),
            ("utf-16-le", "utf-16-le"),
            ("utf-16-be", "utf-16-be"),
        )

    last_error: Exception | None = None
    for load_encoding, save_encoding in encodings:
        try:
            return pysubs2.load(str(path), encoding=load_encoding), save_encoding
        except (UnicodeError, FormatAutodetectionError) as error:
            last_error = error

    if last_error is not None:
        raise last_error
    return pysubs2.load(str(path), encoding="utf-8"), "utf-8"


def load_ass(path: str | Path) -> LoadedSubtitle:
    subtitle_file, encoding = _load_with_encodings(Path(path))
    cues: dict[int, SubtitleCue] = {}
    events_by_id: dict[int, Any] = {}
    cue_id_by_event_id: dict[int, int] = {}
    subtitle_id = 1

    for event_index, event in enumerate(subtitle_file.events):
        if getattr(event, "type", "Dialogue") != "Dialogue":
            continue
        cues[subtitle_id] = SubtitleCue(
            id=subtitle_id,
            event_index=event_index,
            start=event.start / 1000,
            end=event.end / 1000,
            text=event.text,
        )
        events_by_id[subtitle_id] = event
        cue_id_by_event_id[id(event)] = subtitle_id
        subtitle_id += 1

    return LoadedSubtitle(
        file=subtitle_file,
        cues=cues,
        events_by_id=events_by_id,
        cue_id_by_event_id=cue_id_by_event_id,
        encoding=encoding,
    )


def save_ass(document: LoadedSubtitle, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    document.file.save(str(output), encoding=document.encoding)
