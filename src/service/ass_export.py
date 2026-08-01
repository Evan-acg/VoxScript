from __future__ import annotations

from pathlib import Path

from src.entity.subtitle import DialogueLine, NormalizedDialogue, NormalizedSubtitle

_SCRIPT_INFO = "[Script Info]\nScriptType: v4.00+\nWrapStyle: 0\n"
_EVENT_FORMAT = "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"

_DEFAULT_STYLE_FIELDS: dict[str, str] = {
    "Name": "Default",
    "Fontname": "Arial",
    "Fontsize": "20",
    "PrimaryColour": "&H00FFFFFF",
    "SecondaryColour": "&H000000FF",
    "OutlineColour": "&H00000000",
    "BackColour": "&H00000000",
    "Bold": "0",
    "Italic": "0",
    "Underline": "0",
    "StrikeOut": "0",
    "ScaleX": "100",
    "ScaleY": "100",
    "Spacing": "0",
    "Angle": "0",
    "BorderStyle": "1",
    "Outline": "2",
    "Shadow": "0",
    "Alignment": "2",
    "MarginL": "10",
    "MarginR": "10",
    "MarginV": "10",
    "Encoding": "1",
}


def export_ass_file(input_path: Path, output_path: Path) -> Path:
    normalized = NormalizedSubtitle.model_validate_json(
        input_path.read_text(encoding="utf-8")
    )
    output_path.write_text(_render(normalized), encoding="utf-8")
    return output_path


def _render(normalized: NormalizedSubtitle) -> str:
    lines = [
        _SCRIPT_INFO.strip(),
        f"Title: {normalized.path.stem}",
        "",
        _render_styles(normalized.styles),
        "",
        "[Events]",
        _EVENT_FORMAT,
        *(
            _render_dialogue(dialogue)
            for dialogue in normalized.dialogue
            if dialogue.lines
        ),
        "",
    ]
    return "\n".join(lines)


def _render_styles(styles: dict[str, dict[str, str]]) -> str:
    table = styles or {"Default": {}}
    field_names: list[str] = []
    seen: set[str] = set()
    for fields in table.values():
        for key in fields:
            if key not in seen:
                seen.add(key)
                field_names.append(key)
    for key in _DEFAULT_STYLE_FIELDS:
        if key not in seen:
            seen.add(key)
            field_names.append(key)
    if "Name" not in seen:
        field_names.insert(0, "Name")
        seen.add("Name")

    style_lines = ["[V4+ Styles]", f"Format: {', '.join(field_names)}"]
    for name, fields in table.items():
        values = [
            fields.get(key, _DEFAULT_STYLE_FIELDS.get(key, ""))
            for key in field_names
        ]
        values[0] = name
        style_lines.append(f"Style: {', '.join(values)}")
    return "\n".join(style_lines)


def _render_dialogue(dialogue: NormalizedDialogue) -> str:
    style, text = _render_lines(dialogue.lines)
    return (
        f"Dialogue: 0,{_format_ass_time(dialogue.start)},"
        f"{_format_ass_time(dialogue.end)},{style},,0,0,0,,{text}"
    )


def _render_lines(lines: list[DialogueLine]) -> tuple[str, str]:
    style = lines[0].style
    parts: list[str] = []
    current = style
    for line in lines:
        if line.style == current:
            parts.append(line.content)
        else:
            parts.append(f"{{\\r{line.style}}}{line.content}")
            current = line.style
    return style, "\\N".join(parts)


def _format_ass_time(seconds: float) -> str:
    total = round(seconds * 100)
    centis = total % 100
    total //= 100
    secs = total % 60
    total //= 60
    minutes = total % 60
    hours = total // 60
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"
