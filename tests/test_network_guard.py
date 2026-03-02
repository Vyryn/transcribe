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
    results = run_network_guard_self_test()
    assert results["outbound_blocked"] is True
    assert results["loopback_allowed"] is True
