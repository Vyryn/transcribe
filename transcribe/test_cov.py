from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Run the test suite with coverage defaults."""
    try:
        import pytest
    except ImportError:
        print("pytest is not installed. Install dev dependencies first.")
        return 2

    user_args = list(sys.argv[1:] if argv is None else argv)
    args = [
        "tests",
        "--cov=transcribe",
        "--cov-report=term-missing",
        *user_args,
    ]
    return int(pytest.main(args))


if __name__ == "__main__":
    raise SystemExit(main())

