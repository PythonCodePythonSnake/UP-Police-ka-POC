"""Minimal logging setup. Call setup_logging() once at application startup."""

import logging

from app.core.config import settings

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging() -> None:
    level = logging.DEBUG if settings.debug else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT, force=True)

    # Keep third-party HTTP clients quiet; their debug output can include request details.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)