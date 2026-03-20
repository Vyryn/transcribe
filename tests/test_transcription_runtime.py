from __future__ import annotations

import os
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

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


def test_apply_parakeet_runtime_compatibility_disables_live_cuda_graph_decoder_hooks() -> None:
    class FakeGreedyCfg:
        def __init__(self) -> None:
            self.use_cuda_graph_decoder = True

    class FakeCfg:
        def __init__(self) -> None:
            self.decoding = SimpleNamespace(greedy=FakeGreedyCfg())

    class FakeDecoder:
        def __init__(self) -> None:
            self.use_cuda_graph_decoder = True
            self.disable_calls = 0

        def disable_cuda_graphs(self) -> None:
            self.disable_calls += 1
            self.use_cuda_graph_decoder = False

    class FakeModel:
        def __init__(self) -> None:
            self.cfg = FakeCfg()
            self.decoding = FakeDecoder()
            self.disable_calls = 0
            self.change_calls: list[tuple[object, bool]] = []

        def change_decoding_strategy(self, decoding_cfg: object, verbose: bool = True) -> None:
            self.change_calls.append((decoding_cfg, verbose))

        def disable_cuda_graphs(self) -> None:
            self.disable_calls += 1

    model = FakeModel()

    transcription_runtime._apply_parakeet_runtime_compatibility(model, "nvidia/parakeet-tdt-0.6b-v3")

    assert model.cfg.decoding.greedy.use_cuda_graph_decoder is False
    assert model.decoding.use_cuda_graph_decoder is False
    assert model.disable_calls == 1
    assert model.decoding.disable_calls == 1
    assert model.change_calls == [(model.cfg.decoding, False)]


def test_transcribe_row_with_nemo_asr_retries_parakeet_decoder_with_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "chunk.wav"
    audio_path.write_bytes(b"fake-wav")

    class FakeModel:
        def __init__(self) -> None:
            self.calls = 0

        def transcribe(self, paths: list[str]) -> list[dict[str, str]]:
            assert paths == [str(audio_path)]
            self.calls += 1
            if self.calls == 1:
                raise AttributeError("NoneType has no attribute None")
            return [{"text": "hello world"}]

    model = FakeModel()
    observed: list[str] = []

    monkeypatch.setattr(transcription_runtime, "_get_nemo_asr_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        transcription_runtime,
        "_apply_parakeet_runtime_compatibility",
        lambda candidate, resolved_model_id: observed.append(resolved_model_id) or True,
    )
    monkeypatch.setattr(transcription_runtime, "_prepare_nemo_model_for_inference", lambda *args, **kwargs: None)

    text, latency_ms = transcription_runtime.transcribe_row_with_nemo_asr(
        {"audio": {"path": str(audio_path)}},
        "nvidia/parakeet-tdt-0.6b-v3",
    )

    assert text == "hello world"
    assert latency_ms >= 0.0
    assert model.calls == 2
    assert observed == ["nvidia/parakeet-tdt-0.6b-v3"]


def test_get_nemo_asr_model_serializes_concurrent_loads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "parakeet-tdt-0.6b-v3.nemo").write_bytes(b"placeholder")

    fake_nemo = types.ModuleType("nemo")
    fake_collections = types.ModuleType("nemo.collections")
    fake_asr = types.ModuleType("nemo.collections.asr")
    fake_asr.models = SimpleNamespace(ASRModel=object())
    fake_collections.asr = fake_asr
    fake_nemo.collections = fake_collections

    load_started = threading.Event()
    finish_load = threading.Event()
    observed = {"calls": 0}
    sentinel_model = object()

    def fake_restore(nemo_asr: object, resolved_model_id: str, snapshot_dir_arg: str) -> object:
        _ = (nemo_asr, resolved_model_id, snapshot_dir_arg)
        observed["calls"] += 1
        load_started.set()
        assert finish_load.wait(timeout=2.0)
        return sentinel_model

    monkeypatch.setitem(sys.modules, "nemo", fake_nemo)
    monkeypatch.setitem(sys.modules, "nemo.collections", fake_collections)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", fake_asr)
    monkeypatch.setattr(transcription_runtime, "_NEMO_ASR_MODEL_CACHE", {})
    monkeypatch.setattr(transcription_runtime, "_get_hf_repo_snapshot", lambda *args, **kwargs: str(snapshot_dir))
    monkeypatch.setattr(transcription_runtime, "_load_nemo_model_from_snapshot", fake_restore)
    monkeypatch.setattr(
        transcription_runtime,
        "_load_nemo_speechlm_model_from_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("speechlm fallback should not run")),
    )
    monkeypatch.setattr(transcription_runtime, "_apply_parakeet_runtime_compatibility", lambda *args, **kwargs: None)
    monkeypatch.setattr(transcription_runtime, "_prepare_nemo_model_for_inference", lambda *args, **kwargs: None)

    results: list[object] = []
    errors: list[BaseException] = []

    def _worker() -> None:
        try:
            results.append(
                transcription_runtime._get_nemo_asr_model("nvidia/parakeet-tdt-0.6b-v3", local_files_only=True)
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    first = threading.Thread(target=_worker)
    second = threading.Thread(target=_worker)

    first.start()
    assert load_started.wait(timeout=1.0)
    second.start()
    time.sleep(0.1)
    assert observed["calls"] == 1

    finish_load.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert results == [sentinel_model, sentinel_model]
    assert observed["calls"] == 1


def test_transcribe_row_with_granite_asr_decodes_generated_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    class FakeBatch(dict[str, object]):
        def to(self, device: str) -> "FakeBatch":
            observed["batch_device"] = device
            return self

    class FakeProcessor:
        def __call__(self, prompt: str, waveform: object, *, device: str, return_tensors: str) -> FakeBatch:
            observed["prompt"] = prompt
            observed["waveform"] = waveform
            observed["processor_device"] = device
            observed["return_tensors"] = return_tensors
            return FakeBatch({"input_ids": torch.tensor([[11, 12]])})

        def batch_decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> list[str]:
            observed["decoded_ids"] = token_ids.tolist()
            observed["skip_special_tokens"] = skip_special_tokens
            return ["granite transcript"]

    class FakeModel:
        def generate(self, **kwargs: object) -> torch.Tensor:
            observed["generate_kwargs"] = kwargs
            return torch.tensor([[11, 12, 21, 22]])

    runtime = transcription_runtime.GraniteAsrRuntime(
        model=FakeModel(),
        processor=FakeProcessor(),
        device="cpu",
        prompt="USER: <|audio|>\nTranscribe the following speech.\nASSISTANT:",
    )

    monkeypatch.setattr(transcription_runtime, "_get_granite_asr_model", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(transcription_runtime, "_extract_granite_audio_waveform", lambda row: "waveform")

    text, latency_ms = transcription_runtime.transcribe_row_with_granite_asr(
        {"audio": {"bytes": b"fake-wav", "path": "chunk.wav"}},
        "ibm-granite/granite-4.0-1b-speech",
    )

    assert text == "granite transcript"
    assert latency_ms >= 0.0
    assert observed["prompt"] == "USER: <|audio|>\nTranscribe the following speech.\nASSISTANT:"
    assert observed["waveform"] == "waveform"
    assert observed["processor_device"] == "cpu"
    assert observed["batch_device"] == "cpu"
    assert observed["return_tensors"] == "pt"
    assert observed["decoded_ids"] == [[21, 22]]
    assert observed["skip_special_tokens"] is True
