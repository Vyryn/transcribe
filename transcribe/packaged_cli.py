from __future__ import annotations

import os

from transcribe.cli import main as cli_main
from transcribe.runtime_env import PACKAGED_RUNTIME_ENV


def main(argv: list[str] | None = None) -> int:
    os.environ[PACKAGED_RUNTIME_ENV] = "1"
    return cli_main(argv, packaged_runtime=True)
