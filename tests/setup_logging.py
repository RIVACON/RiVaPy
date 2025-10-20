import logging
import os


def setup_logging_for_tests(log_file="rivapy_test.log"):
    """
    Configure logging for unit tests.

    This sets up a global logger for all rivapy.* modules so that
    logs are written both to console (INFO+) and to a test log file (DEBUG+).
    Safe to call multiple times — will not add duplicate handlers.
    """
    logger = logging.getLogger("rivapy")
    logger.setLevel(logging.DEBUG)

    # Prevent re-adding handlers if already configured
    if logger.handlers:
        return logger

    # Ensure log directory exists
    log_path = os.path.abspath(log_file)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # File handler — all logs (DEBUG and above)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    # Console handler — only INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Common format
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add both handlers
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Optional: fine-tune per module
    # logging.getLogger('BIG.marketdata').setLevel(logging.DEBUG)
    # logging.getLogger('BIG.pricing').setLevel(logging.DEBUG)
    # logging.getLogger('BIG.instruments').setLevel(logging.DEBUG)

    return logger
