from pathlib import Path

import yaml

DEFAULT_CONFIG = Path("configs") / "default.yaml"


def load_config(path=DEFAULT_CONFIG, live_mode=None):
    """Read the yaml config. live_mode, if given, overrides the file."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if live_mode is not None:
        cfg["live_mode"] = bool(live_mode)

    return cfg
