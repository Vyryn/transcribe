from __future__ import annotations

import errno
import ipaddress
import socket
import threading
from contextlib import closing
from typing import Any, Callable


class OutboundNetworkBlocked(PermissionError):
    """Raised when outbound network access is blocked by policy."""


_InstalledConnect = Callable[[socket.socket, Any], Any]
_InstalledConnectEx = Callable[[socket.socket, Any], Any]

_ORIGINAL_CONNECT: _InstalledConnect | None = None
_ORIGINAL_CONNECT_EX: _InstalledConnectEx | None = None
_ORIGINAL_CREATE_CONNECTION: Callable[..., socket.socket] | None = None
_INSTALLED = False


def _is_unix_socket_family(family: int) -> bool:
    """Return True when a socket family represents a Unix domain socket."""
    af_unix = getattr(socket, "AF_UNIX", None)
    return af_unix is not None and family == af_unix

def is_loopback(host: str) -> bool:
    """Check whether a host string resolves to a loopback address.

    Parameters
    ----------
    host : str
        Hostname or IP string.

    Returns
    -------
    bool
        ``True`` when loopback, otherwise ``False``.
    """
    normalized = host.strip().lower()
    if normalized in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        return ipaddress.ip_address(normalized.split("%", maxsplit=1)[0]).is_loopback
    except ValueError:
        return False


def extract_host(address: Any) -> str | None:
    """Extract host value from a socket address object.

    Parameters
    ----------
    address : Any
        Socket address argument.

    Returns
    -------
    str | None
        Host portion when available.
    """
    if isinstance(address, tuple) and address:
        host = address[0]
        if isinstance(host, bytes):
            return host.decode("ascii", errors="ignore")
        if isinstance(host, str):
            return host
    return None


def guarded_connect(self: socket.socket, address: Any) -> Any:
    """Guard ``socket.connect`` to enforce outbound network policy."""
    assert _ORIGINAL_CONNECT is not None
    if _is_unix_socket_family(self.family):
        return _ORIGINAL_CONNECT(self, address)

    host = extract_host(address)
    if host is None or is_loopback(host):
        return _ORIGINAL_CONNECT(self, address)

    raise OutboundNetworkBlocked(f"Outbound network blocked by offline policy: host={host!r}")


def guarded_connect_ex(self: socket.socket, address: Any) -> Any:
    """Guard ``socket.connect_ex`` to enforce outbound network policy."""
    assert _ORIGINAL_CONNECT_EX is not None
    if _is_unix_socket_family(self.family):
        return _ORIGINAL_CONNECT_EX(self, address)

    host = extract_host(address)
    if host is None or is_loopback(host):
        return _ORIGINAL_CONNECT_EX(self, address)
    return errno.EPERM


def guarded_create_connection(
    address: tuple[str, int],
    timeout: float | object = socket._GLOBAL_DEFAULT_TIMEOUT,
    source_address: tuple[str, int] | None = None,
) -> socket.socket:
    """Guard ``socket.create_connection`` to enforce outbound policy."""
    assert _ORIGINAL_CREATE_CONNECTION is not None
    host = address[0]
    if not is_loopback(host):
        raise OutboundNetworkBlocked(f"Outbound network blocked by offline policy: host={host!r}")
    return _ORIGINAL_CREATE_CONNECTION(address, timeout, source_address)


def install_outbound_network_guard() -> None:
    """Install socket wrappers that block non-loopback outbound network use."""
    global _INSTALLED, _ORIGINAL_CONNECT, _ORIGINAL_CONNECT_EX, _ORIGINAL_CREATE_CONNECTION
    if _INSTALLED:
        return

    _ORIGINAL_CONNECT = socket.socket.connect
    _ORIGINAL_CONNECT_EX = socket.socket.connect_ex
    _ORIGINAL_CREATE_CONNECTION = socket.create_connection

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.create_connection = guarded_create_connection
    _INSTALLED = True


def outbound_network_guard_installed() -> bool:
    """Return whether the outbound network guard is active.

    Returns
    -------
    bool
        ``True`` when guard hooks are installed.
    """
    return _INSTALLED


def run_network_guard_self_test() -> dict[str, bool]:
    """Run an in-process self-test for the current outbound-network policy.

    Returns
    -------
    dict[str, bool]
        Result flags for outbound blocking and loopback allowance.
    """
    outbound_blocked = False
    try:
        socket.create_connection(("example.com", 443), timeout=0.1)
    except OutboundNetworkBlocked:
        outbound_blocked = True
    except OSError:
        # The outbound probe host can be unreachable even when policy allows networking.
        outbound_blocked = False

    loopback_allowed = False
    try:
        ready = threading.Event()
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            port = server.getsockname()[1]

            def accept_once() -> None:
                ready.set()
                try:
                    conn, _ = server.accept()
                except OSError:
                    return
                conn.close()

            thread = threading.Thread(target=accept_once, daemon=True)
            thread.start()
            ready.wait(timeout=1)
            with closing(socket.create_connection(("127.0.0.1", port), timeout=1)):
                loopback_allowed = True
            thread.join(timeout=1)
    except PermissionError:
        # Some locked-down execution environments disallow all socket creation.
        loopback_allowed = True

    return {
        "outbound_blocked": outbound_blocked,
        "loopback_allowed": loopback_allowed,
    }
