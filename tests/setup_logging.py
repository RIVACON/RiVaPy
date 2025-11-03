import logging
import os
import inspect
import time


class CallerFormatter(logging.Formatter):
    """
    Formatter that adds the external caller (e.g., test function) to each log record.
    """

    _last_time = None
    _start_time = time.time()

    def format(self, record):
        # --- Find external caller ---
        stack = inspect.stack()
        for frame_info in stack[2:]:
            filename = frame_info.filename
            if "tests" in filename:
                record.external_caller = f"{os.path.relpath(filename)}:{frame_info.lineno} in {frame_info.function}()"
                break
        else:
            record.external_caller = "unknown"

        # --- Timing info ---
        now = time.time()
        record.total_elapsed = now - self._start_time
        if self._last_time is None:
            record.delta = 0.0
        else:
            record.delta = now - self._last_time
        self._last_time = now

        return super().format(record)

    # def format(self, record):
    #     # Walk up the stack to find the first frame outside the logging module
    #     stack = inspect.stack()
    #     for frame_info in stack[2:]:
    #         filename = frame_info.filename
    #         # Heuristic: find a file in your tests folder or not in rivapy
    #         if "tests" in filename:
    #             record.external_caller = f"{os.path.relpath(filename)}:{frame_info.lineno} in {frame_info.function}()"
    #             break
    #     else:
    #         record.external_caller = "unknown"
    #     return super().format(record)


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
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")  # does not include callback trace
    # formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s")

    # Formatter includes function and external caller
    # formatter = CallerFormatter(
    #     "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - [called by %(external_caller)s] - %(message)s"
    # )
    # formatter = CallerFormatter(
    #     "%(asctime)s - %(name)s - %(levelname)s - "
    #     "%(filename)s:%(lineno)d - %(funcName)s() - "
    #     "[called by %(external_caller)s] - %(message)s "
    #     "(Δ +%(delta).3fs, total +%(total_elapsed).3fs)"
    # )

    # For callbacks and intense debugging
    # formatter = CallerFormatter(
    #     "%(asctime)s (+%(delta).3fs, total +%(total_elapsed).3fs) - %(name)s - %(levelname)s - "
    #     "%(filename)s:%(lineno)d - %(funcName)s() - "
    #     "[called by %(external_caller)s] - %(message)s"
    # )

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
