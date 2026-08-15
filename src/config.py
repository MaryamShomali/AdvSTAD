import warnings
from pathlib import Path

import torch
import yaml


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return config


def get_device(config):
    device_name = config.get("device")
    if not isinstance(device_name, str):
        raise ValueError("config.yaml must define a string 'device' value")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            "CUDA was requested in config.yaml but is unavailable; using CPU instead.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")
    return device


CONFIG = load_config()
DEVICE = get_device(CONFIG)
