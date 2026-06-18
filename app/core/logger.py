"""
Central logging configuration (only file).

Writes logs into one file:
- path is taken from settings.LOG_FILE_PATH
- default is /app/logs/app.log 

No console logging.
"""

import logging
from pathlib import Path
from .config import settings

def setup_logging() -> None:
    level_name = (settings.LOG_LEVEL or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers (important for reload/import situations)
    if root.handlers:
        return

    log_path = Path(settings.LOG_FILE_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root.addHandler(file_handler)