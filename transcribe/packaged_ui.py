from __future__ import annotations

import tkinter as tk

from transcribe.ui.app import TranscribeUiApp


def main() -> int:
    root = tk.Tk()
    TranscribeUiApp(root, packaged_runtime=True)
    root.mainloop()
    return 0
