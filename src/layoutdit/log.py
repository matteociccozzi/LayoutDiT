# Configure logging singleton
import logging


_logger = None
def get_logger(name: str):
    """
    Get the layout_dit logger singleton.
    """
    global _logger
    
    if _logger is None:
        _logger = logging.getLogger('LayoutDit_' + name)
        if not _logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            _logger.addHandler(handler)
            _logger.setLevel(logging.INFO)
    return _logger
