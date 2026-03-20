from __future__ import annotations

import io
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, cast

SENSITIVE_FIELD_NAMES = {
    "audio",
    "content",
    "input_text",
    "note",
    "payload",
    "raw",
    "text",
    "transcript",
}


class _NullTextStream(io.TextIOBase):
    """Writable text sink used when packaged launches have no console stream."""

    def writable(self) -> bool:
        """Return whether the sink accepts text writes."""
        return True

    def write(self, text: str) -> int:
        """Accept text written by logging handlers and discard it."""
        return len(text)


def _stream_supports_text_writes(stream: object) -> bool:
    """Return whether one candidate stream can accept text output."""
    return stream is not None and not bool(getattr(stream, "closed", False)) and callable(getattr(stream, "write", None))


def resolve_console_stream(*, error: bool, fallback_sink: bool = False) -> io.TextIOBase | None:
    """Resolve a usable stdout/stderr-like text stream.

    Parameters
    ----------
    error : bool
        Whether to prefer stderr instead of stdout.
    fallback_sink : bool, optional
        When ``True``, return a writable null sink if no console stream exists.

    Returns
    -------
    io.TextIOBase | None
        The resolved writable text stream, or ``None`` when no stream exists and
        ``fallback_sink`` is disabled.
    """
    primary = sys.stderr if error else sys.stdout
    fallback = sys.__stderr__ if error else sys.__stdout__
    for candidate in (primary, fallback):
        if _stream_supports_text_writes(candidate):
            return cast(io.TextIOBase, candidate)
    if fallback_sink:
        return _NullTextStream()
    return None


def write_console_line(message: object, *, error: bool = False) -> None:
    """Write one line to an available console stream when present.

    Parameters
    ----------
    message : object
        Message to render.
    error : bool, optional
        Whether to prefer stderr over stdout.
    """
    stream = resolve_console_stream(error=error, fallback_sink=False)
    if stream is None:
        return
    stream.write(f"{message}\n")
    stream.flush()


class JsonFormatter(logging.Formatter):
    """Format log records as structured JSON with optional redaction."""

    def __init__(self, *, redact: bool) -> None:
        """Create a JSON formatter.

        Parameters
        ----------
        redact : bool
            Whether sensitive fields should be redacted.
        """
        super().__init__()
        self._redact = redact

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string.

        Parameters
        ----------
        record : logging.LogRecord
            Logging record to format.

        Returns
        -------
        str
            JSON-encoded log payload.
        """
        payload: dict[str, Any] = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
            "logger": record.name,
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload["fields"] = redact_mapping(fields) if self._redact else fields
        return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def redact_mapping(fields: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive keys from a mapping.

    Parameters
    ----------
    fields : dict[str, Any]
        Mapping to sanitize.

    Returns
    -------
    dict[str, Any]
        Sanitized mapping with sensitive fields replaced.
    """
    redacted: dict[str, Any] = {}
    for key, value in fields.items():
        key_l = key.lower()
        if key_l in SENSITIVE_FIELD_NAMES:
            redacted[key] = "[REDACTED]"
            continue
        if isinstance(value, dict):
            redacted[key] = redact_mapping(value)
            continue
        redacted[key] = value
    return redacted


def configure_logging(level: str = "INFO", *, redact_logs: bool = True) -> None:
    """Configure process-wide structured logging.

    Parameters
    ----------
    level : str, optional
        Root log level name.
    redact_logs : bool, optional
        Whether log fields should be redacted.
    """
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    handler = logging.StreamHandler(resolve_console_stream(error=True, fallback_sink=True))
    handler.setFormatter(JsonFormatter(redact=redact_logs))
    root_logger.addHandler(handler)
    root_logger.setLevel(level.upper())


def security_log(
    logger: logging.Logger,
    level: int,
    event: str,
    **fields: Any,
) -> None:
    """Emit a structured security log event.

    Parameters
    ----------
    logger : logging.Logger
        Logger used to emit the event.
    level : int
        Logging level constant.
    event : str
        Event name.
    **fields : Any
        Structured event metadata.
    """
    logger.log(level, event, extra={"fields": fields})
