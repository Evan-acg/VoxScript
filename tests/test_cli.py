from __future__ import annotations

from click.testing import CliRunner

from src.cli import cli


def test_cli_exposes_only_the_automatic_repair_command() -> None:
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "repair" in result.output
    assert "proof" not in result.output
    assert "trans" not in result.output


def test_repair_help_describes_non_interactive_output() -> None:
    result = CliRunner().invoke(cli, ["repair", "--help"])

    assert result.exit_code == 0
    assert "repaired.ass" in result.output
    assert "chunk-minutes" in result.output
    assert "vad-method" in result.output
    assert "--align" in result.output
