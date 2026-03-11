def main() -> int:
    """Launch the Tk UI lazily."""
    from transcribe.ui.app import main as app_main

    return app_main()


__all__ = ["main"]
