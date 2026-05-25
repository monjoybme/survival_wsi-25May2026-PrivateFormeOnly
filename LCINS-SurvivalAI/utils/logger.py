"""
utils/logger.py
---------------
Configures Python logging to write to both the console and a rotating
log file.  Call `setup_logger()` once at the start of train.py / test.py.
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler


def setup_logger(
    log_dir:   str  = "logs",
    name:      str  = "survival_wsi",
    level:     int  = logging.INFO,
    max_bytes: int  = 10 * 1024 * 1024,  # 10 MB
    backup:    int  = 5,
) -> logging.Logger:
    """
    Create (or retrieve) a logger that writes to console + rotating file.

    Args:
        log_dir:   Directory for the log file.
        name:      Logger name (also used as prefix for the log filename).
        level:     Logging level (INFO, DEBUG, …).
        max_bytes: Max log file size before rotation.
        backup:    Number of backup log files to keep.

    Returns:
        Configured Logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured (e.g., called twice in notebook)
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ───────────────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # ── File handler ──────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = os.path.join(log_dir, f"{name}_{timestamp}.log")
    fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup)
    fh.setLevel(logging.DEBUG)   # file always captures DEBUG and above
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Suppress overly verbose third-party loggers
    for noisy in ("PIL", "openslide", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.info(f"Logger '{name}' initialised — log file: {log_file}")
    return logger
