"""Centralized logging configuration for MGA."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a configured logger for a module.

    Usage:
        logger = get_logger(__name__)

    Args:
        name: Logger name (use __name__ for module logger)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> None:
    """
    Configure root logging for CLI applications.

    Args:
        level: Logging level (logging.INFO, logging.DEBUG, etc.)
        log_file: Optional file path to write logs to
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in handlers:
        handler.setFormatter(fmt)

    root = logging.getLogger("mga")
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in handlers:
        root.addHandler(h)
    root.propagate = False
