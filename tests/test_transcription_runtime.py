from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

import transcribe.transcription_runtime as transcription_runtime


def test_enforce_hf_offline_mode_respects_network_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSCRIBE_ALLOW_NETWORK", "1")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    transcription_runtime._enforce_hf_offline_mode()

    assert os.environ["HF_HUB_OFFLINE"] == "0"
    assert os.environ["HF_DATASETS_OFFLINE"] == "0"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "0"


def test_get_hf_repo_snapshot_uses_network_when_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}
    transcription_runtime._HF_REPO_SNAPSHOT_CACHE.clear()
    monkeypatch.setenv("TRANSCRIBE_ALLOW_NETWORK", "1")
    monkeypatch.setattr(transcription_runtime, "_resolve_packaged_snapshot_path", lambda repo_id: None)

    def fake_snapshot_download(*, repo_id: str, local_files_only: bool) -> str:
        observed["repo_id"] = repo_id
        observed["local_files_only"] = local_files_only
        return "/tmp/transcribe-network-model"

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=fake_snapshot_download))

    snapshot_dir = transcription_runtime._get_hf_repo_snapshot("repo/test-model", local_files_only=True)

    assert snapshot_dir == "/tmp/transcribe-network-model"
    assert observed == {"repo_id": "repo/test-model", "local_files_only": False}
