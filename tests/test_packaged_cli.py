from __future__ import annotations

import pytest


def test_packaged_cli_parser_accepts_models_command() -> None:
    from transcribe.cli import build_parser

    args = build_parser(packaged_runtime=True).parse_args(["models", "list"])

    assert args.command == "models"
    assert args.models_command == "list"


def test_packaged_cli_parser_excludes_bench_commands() -> None:
    from transcribe.cli import build_parser

    parser = build_parser(packaged_runtime=True)
    with pytest.raises(SystemExit):
        parser.parse_args(["bench", "run"])
