from __future__ import annotations

import os
import tkinter as tk

from transcribe.runtime_env import PACKAGED_RUNTIME_ENV
from transcribe.ui.app import TranscribeUiApp


def main() -> int:
    os.environ[PACKAGED_RUNTIME_ENV] = "1"
    root = tk.Tk()
    TranscribeUiApp(root, packaged_runtime=True)
    root.mainloop()
    return 0
