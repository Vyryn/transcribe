from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

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
    handler = logging.StreamHandler()
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
