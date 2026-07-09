"""Centralized logging for the rPPG package.

Library modules obtain a logger via :func:`get_logger` and log through the shared
``rppg`` namespace instead of calling ``print()``. Console output is enabled lazily
the first time a logger is requested, so importing the package never spams stdout, yet
runtime messages (model downloads, weight I/O, dataset scans) remain visible.

The log level is controlled by the ``RPPG_LOG_LEVEL`` environment variable
(e.g. ``DEBUG``, ``INFO``, ``WARNING``); it defaults to ``INFO``.
"""

import logging
import os

_ROOT_NAME = "rppg"
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_configured = False


def _configure_once() -> None:
    """Attach a single console handler to the ``rppg`` root logger (idempotent)."""
    global _configured
    if _configured:
        return

    level_name = os.getenv("RPPG_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(handler)
    # Do not propagate to the global root logger (avoids duplicate lines when the
    # host application also configures logging).
    root.propagate = False

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger (e.g. ``get_logger("detection.mediapipe")``)."""
    _configure_once()
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
