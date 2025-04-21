# Configure logging singleton
import logging
import os


class LayoutDitFilter(logging.Filter):
    def filter(self, record):
        # keep only logs from layout dit
        return record.name.startswith("LayoutDit_")


_logger = None


def get_logger(name: str):
    """
    Get the layout_dit logger singleton.
    """
    global _logger

    if _logger is None:
        _logger = logging.getLogger("LayoutDit_" + name)
        _logger.addFilter(LayoutDitFilter())
        if not _logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            _logger.addHandler(handler)
            _logger.setLevel(logging.INFO)

        level_str = os.getenv("LAYOUT_LOG_LEVEL", "").upper()
        if level_str == "DEBUG":
            _logger.setLevel(logging.DEBUG)
        else:
            _logger.setLevel(logging.INFO)

    return _logger
