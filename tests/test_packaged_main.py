from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "packaged_main.py"


def _load_module() -> object:
    spec = importlib.util.spec_from_file_location("packaged_main", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_packaged_main_launches_gui_without_arguments(monkeypatch) -> None:
    module = _load_module()
    observed: dict[str, object] = {}

    def fake_ui_main() -> int:
        observed["ui_called"] = True
        return 17

    def fake_cli_main(argv: list[str] | None = None) -> int:
        observed["cli_called"] = argv
        return 23

    monkeypatch.setitem(sys.modules, "transcribe.packaged_ui", types.SimpleNamespace(main=fake_ui_main))
    monkeypatch.setitem(sys.modules, "transcribe.packaged_cli", types.SimpleNamespace(main=fake_cli_main))

    result = module.main([])

    assert result == 17
    assert observed == {"ui_called": True}



def test_packaged_main_launches_cli_with_arguments(monkeypatch) -> None:
    module = _load_module()
    observed: dict[str, object] = {}

    def fake_ui_main() -> int:
        observed["ui_called"] = True
        return 17

    def fake_cli_main(argv: list[str] | None = None) -> int:
        observed["cli_called"] = list(argv or [])
        return 23

    monkeypatch.setitem(sys.modules, "transcribe.packaged_ui", types.SimpleNamespace(main=fake_ui_main))
    monkeypatch.setitem(sys.modules, "transcribe.packaged_cli", types.SimpleNamespace(main=fake_cli_main))

    result = module.main(["session", "run"])

    assert result == 23
    assert observed == {"cli_called": ["session", "run"]}
