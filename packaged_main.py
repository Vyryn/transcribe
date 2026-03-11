from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Dispatch packaged launches to the CLI or desktop UI.

    Parameters
    ----------
    argv : list[str] | None, optional
        Command-line arguments excluding the executable name. Uses process argv
        when ``None``.

    Returns
    -------
    int
        Exit code for CLI launches or the GUI process result.
    """
    resolved_argv = list(sys.argv[1:] if argv is None else argv)
    if resolved_argv:
        from transcribe.packaged_cli import main as cli_main

        return cli_main(resolved_argv)

    from transcribe.packaged_ui import main as ui_main

    return ui_main()


if __name__ == "__main__":
    raise SystemExit(main())
