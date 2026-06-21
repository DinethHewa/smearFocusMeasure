"""Reusable logging helpers for the corrected scaffold."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    *,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """Return a configured logger without duplicating handlers."""

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    config_token = (
        str(Path(log_file).resolve()) if log_file is not None else None,
        level,
        console,
    )
    if getattr(logger, "_focus_measure_logger_token", None) == config_token:
        return logger

    logger.handlers.clear()
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if console:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger._focus_measure_logger_token = config_token  # type: ignore[attr-defined]
    return logger


__all__ = ["get_logger"]

