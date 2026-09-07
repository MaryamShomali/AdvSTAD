import copy
import io

import numpy as np
import pytest
import torch

from src.advstad_checkpoint import (
    ALIGNMENT_POLICY,
    checkpoint_metadata,
    checkpoint_payload,
    experiment_key,
    restore_checkpoint,
)
from src.advstad_training import adversarial_coefficient, build_optimizers, evaluate, train_epoch
from src.models import AdvSTAD


def make_model(fusion="sum", dtype=torch.float32, **overrides):
    settings = {
        "window_size": 4, "batch_size": 2, "d_model": 12, "nhead": 3,
        "dropout": 0.0, "fusion": fusion,
        "training": {"learning_rate": 0.001},
    }
    settings.update(overrides)
    return AdvSTAD(3, settings).to(dtype=dtype)


def trained_payload(fusion="sum"):
    torch.manual_seed(42)
    model = make_model(fusion)
    optimizers, schedulers = build_optimizers(model)
    data = torch.rand(3, model.n_window, model.n_feats)
    metrics = train_epoch(model, data, optimizers, schedulers, epoch=2)
    model.training_history = [{"epoch": 2, **metrics}]
    accuracy = [(metrics["loss_g"], metrics["lr_g"])]
    buffer = io.BytesIO()
    torch.save(checkpoint_payload(model, optimizers, schedulers, 2, accuracy), buffer)
    buffer.seek(0)
    # Compatible with the repository's PyTorch 1.8 and newer weights-only loaders.
    payload = torch.load(buffer, map_location="cpu")
    return model, optimizers, schedulers, data, payload


def assert_nested_equal(actual, expected):
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            assert_nested_equal(value, reference)
    else:
        assert actual == expected


@pytest.mark.parametrize("fusion", ["sum", "concat", "cross_attention"])
def test_round_trip_predictions_optimizer_scheduler_history_and_absolute_epoch(fusion):
    original, original_opts, original_schedulers, data, payload = trained_payload(fusion)
    restored = make_model(fusion)
    optimizers, schedulers = build_optimizers(restored)
    epoch, accuracy = restore_checkpoint(payload, restored, optimizers, schedulers)

    assert epoch == 2
    assert accuracy == payload["accuracy_list"]
    assert restored.training_history == original.training_history
    assert restored.training_history is not payload["training_history"]
    for actual, expected in zip(evaluate(restored, data), evaluate(original, data)):
        np.testing.assert_array_equal(actual, expected)
    for name in optimizers:
        assert_nested_equal(optimizers[name].state_dict(), original_opts[name].state_dict())
        assert_nested_equal(schedulers[name].state_dict(), original_schedulers[name].state_dict())
    assert adversarial_coefficient(restored, epoch + 1) == pytest.approx(0.075)

    # A resumed update follows the same trajectory, including both optimizer moments.
    continued = train_epoch(original, data, original_opts, original_schedulers, epoch + 1)
    resumed = train_epoch(restored, data, optimizers, schedulers, epoch + 1)
    assert continued == pytest.approx(resumed)
    for actual, expected in zip(restored.parameters(), original.parameters()):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"fusion": "concat"}, "fusion"),
        ({"nhead": 2}, "nhead"),  # All state tensor shapes still match.
        ({"d_model": 6}, "d_model"),
        ({"window_size": 5}, "window_size"),
        ({"training": {"learning_rate": 0.001, "adversarial_weight": 0.2}}, "adversarial_weight"),
    ],
)
def test_reject_semantic_mismatch_before_mutating_weights(overrides, reason):
    original = make_model()
    opts, scheds = build_optimizers(original)
    payload = checkpoint_payload(original, opts, scheds, -1, [])
    target = make_model(**overrides)
    original_target_state = copy.deepcopy(target.state_dict())
    opts, scheds = build_optimizers(target)
    with pytest.raises(ValueError, match=reason):
        restore_checkpoint(payload, target, opts, scheds)
    assert_nested_equal(target.state_dict(), original_target_state)
    assert all(not optimizer.state for optimizer in opts.values())


@pytest.mark.parametrize("metadata_field", ["window_alignment", "objective_version", "schedule_version"])
def test_reject_incompatible_alignment_and_objective_versions(metadata_field):
    model = make_model()
    opts, scheds = build_optimizers(model)
    payload = checkpoint_payload(model, opts, scheds, -1, [])
    payload["metadata"][metadata_field] = "different_version"
    with pytest.raises(ValueError, match=metadata_field):
        restore_checkpoint(payload, model, opts, scheds)


def test_experiment_identity_tracks_semantics_but_not_runtime_or_invocation_length():
    model = make_model()
    reference = experiment_key(model, "SMD")
    assert reference.startswith("AdvSTAD_SMD_sum_")
    assert experiment_key(make_model(), "SMD") == reference
    model.to(dtype=torch.float64)
    model.run_name = "another-label"
    model.resolved_config["training"]["epochs_per_run"] = 37
    assert experiment_key(model, "SMD") == reference
    assert experiment_key(model, "synthetic") != reference
    assert experiment_key(make_model(fusion="concat"), "SMD") != reference
    assert experiment_key(make_model(nhead=2), "SMD") != reference
    assert experiment_key(make_model(window_size=5), "SMD") != reference
    different_training = make_model(training={"learning_rate": 0.001, "adversarial_weight": 0.0})
    assert experiment_key(different_training, "SMD") != reference

    original = make_model()
    opts, scheds = build_optimizers(original)
    payload = checkpoint_payload(original, opts, scheds, 6, [])
    target_opts, target_scheds = build_optimizers(model)
    epoch, _ = restore_checkpoint(payload, model, target_opts, target_scheds)
    assert epoch == 6
    assert model.resolved_config["training"]["epochs_per_run"] == 37


def test_sensor_identity_metadata_validates_column_order():
    model = make_model()
    model.sensor_identifiers = ["temperature", "pressure", "flow"]
    assert checkpoint_metadata(model)["sensor_identifiers"] == model.sensor_identifiers
    assert checkpoint_metadata(model)["window_alignment"] == ALIGNMENT_POLICY
    opts, scheds = build_optimizers(model)
    payload = checkpoint_payload(model, opts, scheds, -1, [])
    model.sensor_identifiers = list(reversed(model.sensor_identifiers))
    with pytest.raises(ValueError, match="sensor_identifiers"):
        restore_checkpoint(payload, model, opts, scheds)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
))])
def test_optimizer_state_migrates_to_parameter_device_and_dtype(device):
    _, _, _, data, payload = trained_payload()
    target = make_model(dtype=torch.float64).to(device=device)
    opts, scheds = build_optimizers(target)
    epoch, _ = restore_checkpoint(payload, target, opts, scheds)
    for optimizer in opts.values():
        assert optimizer.state
        for parameter, state in optimizer.state.items():
            for value in state.values():
                if torch.is_tensor(value):
                    assert value.device == parameter.device
                    if value.is_floating_point():
                        assert value.dtype == parameter.dtype
    # Device/dtype conversion must leave both optimizers usable.
    metrics = train_epoch(target, data, opts, scheds, epoch + 1)
    assert all(np.isfinite(value) for value in metrics.values())


def test_reject_legacy_payload_and_missing_owner_states():
    model = make_model()
    opts, scheds = build_optimizers(model)
    with pytest.raises(ValueError, match="legacy TranAD"):
        restore_checkpoint({"model_state_dict": model.state_dict()}, model, opts, scheds)
    payload = checkpoint_payload(model, opts, scheds, -1, [])
    del payload["optimizer_state_dicts"]["adversary"]
    with pytest.raises(ValueError, match="generator and adversary"):
        restore_checkpoint(payload, model, opts, scheds)


def test_reject_incorrect_state_tensor_shape_before_mutation():
    model = make_model()
    opts, scheds = build_optimizers(model)
    payload = checkpoint_payload(model, opts, scheds, -1, [])
    state_name = next(name for name, tensor in model.state_dict().items() if tensor.numel() > 1)
    payload["model_state_dict"][state_name] = torch.zeros(1)
    target = make_model()
    before = copy.deepcopy(target.state_dict())
    opts, scheds = build_optimizers(target)
    with pytest.raises(ValueError, match="different shape"):
        restore_checkpoint(payload, target, opts, scheds)
    assert_nested_equal(target.state_dict(), before)
