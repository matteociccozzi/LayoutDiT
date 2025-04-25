import json

import fsspec

from layoutdit.configuration.config_constructs import LayoutDitConfig

_layout_dit_config = None


def read_config_from_gcs_if_exists() -> LayoutDitConfig | None:
    path = "gs://layoutdit/layout_dit_config.json"
    fs = fsspec.filesystem("gcs")

    if not fs.exists(path):
        return None

    with fs.open(path, "r") as f:
        config_dict = json.load(f)

    return LayoutDitConfig(**config_dict)


def get_layout_dit_config():
    """
    Get the LayoutDitConfig singleton.
    """
    global _layout_dit_config

    if _layout_dit_config is None:
        _layout_dit_config = read_config_from_gcs_if_exists()
        if _layout_dit_config is None:  # use default
            _layout_dit_config = LayoutDitConfig()

    return _layout_dit_config
