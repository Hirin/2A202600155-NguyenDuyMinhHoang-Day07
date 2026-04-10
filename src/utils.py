"""Logging utilities — ghi đồng thời ra stdout và logs/log.txt."""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(name: str = "rag") -> logging.Logger:
    """Trả về logger ghi đồng thời ra console và logs/log.txt."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "log.txt"

    logger = logging.getLogger(name)
    if logger.handlers:          # Tránh thêm handler trùng lặp
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # Handler ghi ra file (append)
    fh = logging.FileHandler(log_file, encoding="utf-8", mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Handler ghi ra stdout
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    # In dấu phân cách mỗi lần chạy mới
    separator = f"\n{'='*70}\nRUN STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*70}"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(separator + "\n")

    return logger
