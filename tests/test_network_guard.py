from __future__ import annotations

import socket

import pytest

from transcribe.network_guard import (
    OutboundNetworkBlocked,
    install_outbound_network_guard,
    outbound_network_guard_installed,
    run_network_guard_self_test,
)


def test_install_outbound_network_guard_idempotent() -> None:
    install_outbound_network_guard()
    install_outbound_network_guard()
    assert outbound_network_guard_installed() is True


def test_outbound_network_is_blocked() -> None:
    install_outbound_network_guard()
    with pytest.raises(OutboundNetworkBlocked):
        socket.create_connection(("example.com", 443), timeout=0.1)


def test_loopback_remains_allowed() -> None:
    install_outbound_network_guard()
    results = run_network_guard_self_test()
    assert results["outbound_blocked"] is True
    assert results["loopback_allowed"] is True


def test_self_test_reflects_current_process_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeConnection:
        def close(self) -> None:
            return None

    def fake_create_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None):
        del timeout, source_address
        host = address[0]
        calls.append(host)
        if host == "example.com":
            raise OSError("network path is open but the probe host is unavailable")
        return FakeConnection()

    monkeypatch.setattr(socket, "create_connection", fake_create_connection)

    results = run_network_guard_self_test()

    assert results["outbound_blocked"] is False
    assert results["loopback_allowed"] is True
    assert calls[:2] == ["example.com", "127.0.0.1"]
