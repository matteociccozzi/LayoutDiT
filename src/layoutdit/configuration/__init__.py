from layoutdit.configuration.config_constructs import LayoutDitConfig

_layout_dit_config = None


def get_layout_dit_config():
    """
    Get the LayoutDitConfig singleton.
    """
    global _layout_dit_config

    if _layout_dit_config is None:
        _layout_dit_config = LayoutDitConfig()

    return _layout_dit_config
