"""
logger.py

Centralized logging configuration for the project.

Why this file exists:
    Using print() statements for debugging becomes unmanageable once the
    project grows (dataset loading, training loops, real-time detection).
    Python's built-in `logging` module lets us:
        1. Timestamp every message automatically.
        2. Write to BOTH the console and a log file simultaneously.
        3. Categorize messages by severity (INFO, WARNING, ERROR, etc.)
        4. Easily silence or filter logs later without touching every file.

    Every other module in this project will call get_logger(__name__)
    to get a consistently configured logger.
"""

import logging
from pathlib import Path

from src.config import LOGS_DIR, ensure_directories_exist


def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a configured logger instance.

    Args:
        name: The name of the logger, conventionally passed as __name__
              from the calling module. This makes log messages show
              WHICH file they came from (e.g., "src.dataset").

    Returns:
        A logging.Logger instance configured to write to both the
        console and a shared log file (logs/project.log).
    """
    # Make sure logs/ folder exists before we try to write to it.
    ensure_directories_exist()

    logger = logging.getLogger(name)

    # Guard against adding duplicate handlers.
    # Why this matters: if get_logger(__name__) is called multiple times
    # for the same module (e.g., during re-imports in notebooks), we don't
    # want the same log message to be printed multiple times.
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)  # Capture everything; handlers below filter what's shown.

    # ---------------------------------------------------------------
    # FORMATTER: defines what each log line looks like
    # ---------------------------------------------------------------
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ---------------------------------------------------------------
    # CONSOLE HANDLER: prints logs to your terminal
    # ---------------------------------------------------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Show INFO and above in console
    console_handler.setFormatter(formatter)

    # ---------------------------------------------------------------
    # FILE HANDLER: writes logs to logs/project.log
    # ---------------------------------------------------------------
    log_file_path: Path = LOGS_DIR / "project.log"
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # Log EVERYTHING (including DEBUG) to file
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    # Quick manual test: run `python src/logger.py` directly.
    test_logger = get_logger(__name__)
    test_logger.debug("This is a DEBUG message (only visible in the log file).")
    test_logger.info("This is an INFO message (visible in console and file).")
    test_logger.warning("This is a WARNING message.")
    test_logger.error("This is an ERROR message.")
    print(f"\nCheck the log file at: {LOGS_DIR / 'project.log'}")