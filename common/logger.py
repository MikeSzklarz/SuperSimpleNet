import logging
import sys
from pathlib import Path


def setup_logger(log_path: Path | None = None, name: str = "ssn") -> logging.Logger:
    """Set up the named logger with a console handler and optional file handler.

    Args:
        log_path: directory to write training.log; if None, file logging is skipped
        name: logger name used throughout the codebase via logging.getLogger(name)
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)

    if log_path is not None:
        log_path.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path / "training.log", mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)

    return logger


def add_file_handler(log_dir: Path) -> None:
    """Attach a file handler to the 'ssn' logger after it has been set up."""
    logger = logging.getLogger("ssn")
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "training.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)


def get_logger() -> logging.Logger:
    return logging.getLogger("ssn")
