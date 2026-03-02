from __future__ import annotations

import json
import logging

from transcribe.logging import configure_logging, security_log


def test_security_log_redacts_sensitive_payload(capsys) -> None:
    configure_logging("INFO", redact_logs=True)
    logger = logging.getLogger("transcribe-test")
    security_log(
        logger,
        logging.INFO,
        "capture_event",
        transcript="very sensitive transcript",
        pair_count=10,
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.err.strip())
    assert payload["fields"]["transcript"] == "[REDACTED]"
    assert payload["fields"]["pair_count"] == 10
