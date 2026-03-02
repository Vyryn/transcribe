from __future__ import annotations

import json
from pathlib import Path

from transcribe.models import SessionManifest


def write_session_manifest(manifest: SessionManifest, path: Path) -> None:
    """Write a session manifest as UTF-8 JSON.

    Parameters
    ----------
    manifest : SessionManifest
        Manifest object to serialize.
    path : Path
        Output path for serialized JSON manifest.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
