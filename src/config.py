import warnings
import copy
import math
from collections.abc import Mapping
from numbers import Integral, Real
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


def get_dtype(config):
    dtype_name = config.get("dtype", "float64")
    dtypes = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if dtype_name not in dtypes:
        supported = ", ".join(dtypes)
        raise ValueError(f"config.yaml 'dtype' must be one of: {supported}")
    return dtypes[dtype_name]


ADVSTAD_DEFAULTS = {
    "window_size": 10,
    "batch_size": 128,
    "d_model": None,
    "nhead": None,
    "temporal_layers": 1,
    "spatial_layers": 1,
    "decoder_layers": 1,
    "dim_feedforward": 16,
    "dropout": 0.1,
    "fusion": "sum",
    "training": {
        "epochs_per_run": 5,
        "learning_rate": None,
        "adversarial_weight": 0.1,
        "gradient_clip_norm": 1.0,
    },
}


def _positive_integer(value, path):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{path} must be a positive integer (not a boolean)")
    return int(value)


def _finite_number(value, path):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{path} must be a finite number")
    return float(value)


def get_advstad_config(config, feats, default_lr):
    """Resolve only AdvSTAD settings, without changing the supplied root mapping."""
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    feats = _positive_integer(feats, "AdvSTAD feature count")
    supplied = config.get("advstad", {})
    if not isinstance(supplied, Mapping):
        raise ValueError("advstad must be a mapping")
    unknown = set(supplied) - set(ADVSTAD_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown advstad keys: {', '.join(sorted(map(str, unknown)))}")
    training = supplied.get("training", {})
    if not isinstance(training, Mapping):
        raise ValueError("advstad.training must be a mapping")
    unknown = set(training) - set(ADVSTAD_DEFAULTS["training"])
    if unknown:
        raise ValueError(f"Unknown advstad.training keys: {', '.join(sorted(map(str, unknown)))}")

    resolved = copy.deepcopy(ADVSTAD_DEFAULTS)
    resolved.update({key: value for key, value in supplied.items() if key != "training"})
    resolved["training"].update(training)
    if resolved["d_model"] is None:
        resolved["d_model"] = 2 * feats
    if resolved["nhead"] is None:
        resolved["nhead"] = feats
    for key in (
        "window_size", "batch_size", "d_model", "nhead", "temporal_layers",
        "spatial_layers", "decoder_layers", "dim_feedforward",
    ):
        resolved[key] = _positive_integer(resolved[key], f"advstad.{key}")
    if resolved["d_model"] % resolved["nhead"]:
        raise ValueError("advstad.d_model must be divisible by advstad.nhead")
    if resolved["fusion"] not in ("sum", "concat", "cross_attention"):
        raise ValueError("advstad.fusion must be exactly sum, concat, or cross_attention")
    resolved["dropout"] = _finite_number(resolved["dropout"], "advstad.dropout")
    if not 0 <= resolved["dropout"] < 1:
        raise ValueError("advstad.dropout must be in [0, 1)")

    training = resolved["training"]
    training["epochs_per_run"] = _positive_integer(
        training["epochs_per_run"], "advstad.training.epochs_per_run"
    )
    if training["learning_rate"] is None:
        training["learning_rate"] = default_lr
    for key in ("learning_rate", "gradient_clip_norm", "adversarial_weight"):
        if key == "gradient_clip_norm" and training[key] is None:
            continue
        training[key] = _finite_number(training[key], f"advstad.training.{key}")
        if key == "adversarial_weight":
            if not 0 <= training[key] <= 1:
                raise ValueError("advstad.training.adversarial_weight must be in [0, 1]")
        elif training[key] <= 0:
            raise ValueError(f"advstad.training.{key} must be positive")
    return resolved


CONFIG = load_config()
DEVICE = get_device(CONFIG)
DTYPE = get_dtype(CONFIG)
