from __future__ import annotations

from pathlib import Path

from transcribe.compliance import enforce_no_url_literals


def main() -> int:
    """Run URL literal compliance check from current working directory.

    Returns
    -------
    int
        Exit code where ``0`` indicates success.
    """
    return enforce_no_url_literals(Path.cwd())


if __name__ == "__main__":
    raise SystemExit(main())
