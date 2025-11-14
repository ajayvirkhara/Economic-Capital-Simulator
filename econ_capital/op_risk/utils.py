import logging
import time
from contextlib import contextmanager


def setup_logging(name: str = "op_risk"):
    """Standardized logger setup."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@contextmanager
def timed_section(section_name: str):
    """Context manager to measure execution time."""
    logger = logging.getLogger("op_risk")  # Ensure logs go through standard logger
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info("[%s] completed in %.2f s", section_name, elapsed)
