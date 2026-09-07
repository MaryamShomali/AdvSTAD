"""Versioned persistence and shared artifact identifiers for AdvSTAD only."""

import copy
import hashlib
import json
import re
from collections.abc import Mapping

import torch


SCHEMA_VERSION = 1
ALIGNMENT_POLICY = "inclusive_endpoint_v1"
OBJECTIVE_VERSION = "anchored_adversarial_v1"
SCHEDULE_VERSION = "absolute_epoch_growth_v1"
_OWNERS = {"generator", "adversary"}


def checkpoint_metadata(model):
    """Describe semantics that tensor shapes alone cannot validate."""
    config = copy.deepcopy(model.resolved_config)
    identifiers = getattr(model, "sensor_identifiers", None)
    if identifiers is None:
        identifiers = getattr(model, "sensor_order", None)
    if identifiers is not None:
        if isinstance(identifiers, (str, bytes)):
            raise ValueError("AdvSTAD sensor identifiers must be an ordered sequence")
        identifiers = list(identifiers)
        if len(identifiers) != model.n_feats:
            raise ValueError("AdvSTAD sensor identifiers must match the feature count")
    return {
        "model": "AdvSTAD",
        "resolved_config": config,
        "n_feats": model.n_feats,
        "n_window": model.n_window,
        "sensor_identifiers": identifiers,
        "window_alignment": ALIGNMENT_POLICY,
        "objective_version": OBJECTIVE_VERSION,
        "schedule_version": SCHEDULE_VERSION,
    }


def _semantics(metadata):
    semantics = copy.deepcopy(metadata)
    # Duration of this invocation does not restart the absolute epoch schedule.
    semantics["resolved_config"]["training"].pop("epochs_per_run", None)
    return semantics


def experiment_key(model, dataset):
    """Use the same stable key for checkpoints, plots, and tracked artifacts."""
    semantics = _semantics(checkpoint_metadata(model))
    serialized = json.dumps(semantics, sort_keys=True, separators=(",", ":"), allow_nan=False)
    fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
    dataset_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(dataset)).strip("._") or "dataset"
    fusion = model.resolved_config["fusion"]
    return f"AdvSTAD_{dataset_name}_{fusion}_{fingerprint}"


def _validate_owners(collection, description):
    if not isinstance(collection, Mapping) or set(collection) != _OWNERS:
        raise ValueError(f"AdvSTAD {description} must have generator and adversary entries")


def checkpoint_payload(model, optimizers, schedulers, epoch, accuracy_list):
    """Build the AdvSTAD payload; the caller performs atomic file replacement."""
    _validate_owners(optimizers, "optimizers")
    _validate_owners(schedulers, "schedulers")
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": checkpoint_metadata(model),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dicts": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        "scheduler_state_dicts": {name: scheduler.state_dict() for name, scheduler in schedulers.items()},
        "epoch": epoch,
        "accuracy_list": copy.deepcopy(accuracy_list),
        "training_history": copy.deepcopy(getattr(model, "training_history", [])),
    }


def _incompatible(reason):
    raise ValueError(
        f"Incompatible AdvSTAD checkpoint: {reason}. "
        "Use the matching AdvSTAD configuration and sensor order, or start a new run with --retrain."
    )


def _differences(saved, expected, prefix="metadata"):
    if isinstance(saved, Mapping) and isinstance(expected, Mapping):
        differences = []
        for key in sorted(set(saved) | set(expected)):
            path = f"{prefix}.{key}"
            if key not in saved or key not in expected:
                differences.append(path)
            else:
                differences.extend(_differences(saved[key], expected[key], path))
        return differences
    return [] if saved == expected else [prefix]


def _validate_checkpoint(checkpoint, model, optimizers, schedulers):
    # All metadata and structural checks run before copying any model weights.
    if not isinstance(checkpoint, Mapping) or checkpoint.get("schema_version") != SCHEMA_VERSION:
        _incompatible(f"expected schema_version={SCHEMA_VERSION}; legacy TranAD weights cannot be converted")
    required = {
        "metadata", "model_state_dict", "optimizer_state_dicts", "scheduler_state_dicts",
        "epoch", "accuracy_list", "training_history",
    }
    missing = required - set(checkpoint)
    if missing:
        _incompatible(f"missing fields {sorted(missing)}")
    metadata = checkpoint["metadata"]
    if not isinstance(metadata, Mapping) or not isinstance(metadata.get("resolved_config"), Mapping):
        _incompatible("metadata.resolved_config must be a mapping")
    if not isinstance(metadata["resolved_config"].get("training"), Mapping):
        _incompatible("metadata.resolved_config.training must be a mapping")
    differences = _differences(_semantics(metadata), _semantics(checkpoint_metadata(model)))
    if differences:
        _incompatible("different " + ", ".join(differences))
    epoch = checkpoint["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < -1:
        _incompatible("epoch must be the completed zero-based epoch, or -1")
    for field in ("accuracy_list", "training_history"):
        if not isinstance(checkpoint[field], list):
            _incompatible(f"{field} must be a list")

    saved_state = checkpoint["model_state_dict"]
    current_state = model.state_dict()
    if not isinstance(saved_state, Mapping) or set(saved_state) != set(current_state):
        _incompatible("model state keys do not match")
    for name, tensor in current_state.items():
        saved_tensor = saved_state[name]
        if not torch.is_tensor(saved_tensor) or saved_tensor.shape != tensor.shape:
            _incompatible(f"model tensor {name} has a different shape")
    for field, objects in (("optimizer_state_dicts", optimizers), ("scheduler_state_dicts", schedulers)):
        states = checkpoint[field]
        if not isinstance(states, Mapping) or set(states) != _OWNERS:
            _incompatible(f"{field} must contain generator and adversary states")
        for name in objects:
            if not isinstance(states[name], Mapping):
                _incompatible(f"{field}.{name} must be a mapping")
    for name, optimizer in optimizers.items():
        state = checkpoint["optimizer_state_dicts"][name]
        groups = state.get("param_groups")
        if not isinstance(state.get("state"), Mapping) or not isinstance(groups, list):
            _incompatible(f"invalid optimizer state for {name}")
        expected_groups = optimizer.state_dict()["param_groups"]
        if len(groups) != len(expected_groups):
            _incompatible(f"different optimizer parameter groups for {name}")
        for saved, expected in zip(groups, expected_groups):
            if not isinstance(saved, Mapping) or not isinstance(saved.get("params"), list):
                _incompatible(f"invalid optimizer parameter group for {name}")
            if len(saved["params"]) != len(expected["params"]):
                _incompatible(f"different optimizer parameter counts for {name}")


def _move_state(value, parameter):
    if torch.is_tensor(value):
        dtype = parameter.dtype if value.is_floating_point() else value.dtype
        return value.to(device=parameter.device, dtype=dtype)
    if isinstance(value, dict):
        return {key: _move_state(item, parameter) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_state(item, parameter) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_state(item, parameter) for item in value)
    return value


def restore_checkpoint(checkpoint, model, optimizers, schedulers):
    """Validate compatibility, restore both owners, and resume absolute epochs."""
    _validate_owners(optimizers, "optimizers")
    _validate_owners(schedulers, "schedulers")
    _validate_checkpoint(checkpoint, model, optimizers, schedulers)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(checkpoint["optimizer_state_dicts"][name])
        for parameter, state in optimizer.state.items():
            optimizer.state[parameter] = _move_state(state, parameter)
        schedulers[name].load_state_dict(checkpoint["scheduler_state_dicts"][name])
    model.training_history = copy.deepcopy(checkpoint["training_history"])
    return checkpoint["epoch"], copy.deepcopy(checkpoint["accuracy_list"])
