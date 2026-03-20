from __future__ import annotations

from transcribe.cli import main as cli_main


def main(argv: list[str] | None = None) -> int:
    return cli_main(argv, packaged_runtime=True)
