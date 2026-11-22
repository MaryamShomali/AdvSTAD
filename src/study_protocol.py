"""Shared study identifiers; safe to import without loading models or CLI parsing."""

import hashlib
from pathlib import Path


ABLATION_MODELS = ("TranAD", "TranAD_Corrected", "AdvSTAD_NoSpatial", "AdvSTAD")
ADVSTAD_MODELS = ("AdvSTAD", "AdvSTAD_NoSpatial")
SMD_MACHINE_ID = "machine-1-1"
DEFAULT_SEED = 1
INCLUSIVE_ALIGNMENT = "inclusive_endpoint_v1"
LEGACY_ALIGNMENT = "legacy_previous_endpoint"
TIMING_VERSION = "synchronized_perf_counter_v1"


def window_alignment(model_name):
    if model_name in ADVSTAD_MODELS or model_name == "TranAD_Corrected":
        return INCLUSIVE_ALIGNMENT
    if "TranAD" in model_name:
        return LEGACY_ALIGNMENT
    return None


def dataset_provenance(dataset, processed_root="processed"):
    """Fingerprint the same single-machine arrays used by the existing loader.

    File hashing is outside all training/inference timing boundaries. No data is
    transformed, and no other SMD machine is read.
    """
    if dataset != "SMD":
        return {}
    sources, fingerprints = {}, {}
    for split in ("train", "test", "labels"):
        path = Path(processed_root) / dataset / f"{SMD_MACHINE_ID}_{split}.npy"
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        sources[split] = str(path)
        fingerprints[split] = digest.hexdigest()
    return {
        "machine_id": SMD_MACHINE_ID,
        "source_files": sources,
        "data_fingerprints": fingerprints,
    }


def evaluation_protocol(level, multiplier):
    """Describe existing scoring and metrics without changing their definitions."""
    return {
        "version": "legacy_pot_point_adjusted_v1",
        "score": "decoder2_endpoint_squared_error",
        "aggregation": "mean_across_sensors",
        "labels": "any_positive_sensor",
        "point_adjustment": True,
        "roc_auc_input": "adjusted_binary_predictions",
        "pot": {
            "q": 1e-5,
            "level": level,
            "multiplier": multiplier,
            "dynamic": False,
            "max_attempts": 10,
        },
    }
