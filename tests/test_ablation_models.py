"""Tiny synthetic checks for the architecture and checkpoint ablations."""

import copy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.advstad_checkpoint import (
    checkpoint_metadata,
    checkpoint_payload,
    experiment_key,
    restore_checkpoint,
)
from src.advstad_training import (
    build_optimizers,
    evaluate,
    objectives,
    parameter_groups,
    train_epoch,
)
from src.models import AdvSTAD, AdvSTAD_NoSpatial, TranAD, TranAD_Corrected


FUSIONS = ("sum", "concat", "cross_attention")


def settings(fusion="sum", **overrides):
    result = {
        "window_size": 4, "batch_size": 2, "d_model": 8, "nhead": 2,
        "dropout": 0.0, "fusion": fusion,
        "training": {"learning_rate": 0.001},
    }
    result.update(overrides)
    return result


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def test_corrected_tranad_preserves_architecture_initialization_and_state_compatibility():
    torch.manual_seed(93)
    original = TranAD(3).eval()
    state_after_original = torch.get_rng_state()
    torch.manual_seed(93)
    corrected = TranAD_Corrected(3).eval()
    assert corrected.name == "TranAD_Corrected"
    assert original.name == "TranAD"
    assert torch.equal(torch.get_rng_state(), state_after_original)
    assert original.state_dict().keys() == corrected.state_dict().keys()
    assert parameter_count(original) == parameter_count(corrected)
    for name, value in original.state_dict().items():
        assert torch.equal(value, corrected.state_dict()[name])
    corrected.load_state_dict(original.state_dict(), strict=True)
    src = torch.rand(original.n_window, 2, 3)
    target = src[-1:]
    expected, actual = original(src, target), corrected(src, target)
    for original_output, corrected_output in zip(expected, actual):
        assert original_output.shape == corrected_output.shape == (1, 2, 3)
        assert torch.equal(original_output, corrected_output)
    sum((value - target).square().mean() for value in actual).backward()
    assert all(parameter.grad is not None for parameter in corrected.parameters())


@pytest.mark.parametrize("fusion", FUSIONS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("width,heads", [(6, 3), (8, 2), (5, 1)])
def test_temporal_only_forward_losses_and_backpropagation(fusion, dtype, width, heads):
    model = AdvSTAD_NoSpatial(3, settings(fusion, d_model=width, nhead=heads)).to(dtype=dtype)
    src = torch.rand(4, 2, 3, dtype=dtype)
    target = src[-1:]
    predictions = model(src, target, return_aux=True)
    assert model.name == "AdvSTAD_NoSpatial"
    assert predictions.keys() == {"y1", "y2", "y2_base"}
    for prediction in predictions.values():
        assert prediction.shape == (1, 2, 3)
        assert prediction.dtype == dtype
        assert torch.isfinite(prediction).all()
    losses = objectives(target=target, alpha=0.05, **predictions)
    assert all(torch.isfinite(value) for value in losses.values())
    # Both objectives use the unchanged AdvSTAD formulation. Summing the three
    # reconstruction errors exercises both decoders and the attached conditioning.
    sum(losses[name] for name in ("error_y1", "error_y2_base", "error_y2")).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_no_spatial_modules_are_never_constructed_and_temporal_memory_is_unprojected():
    # This also catches implementations that construct then delete a fusion path.
    with patch("src.models.SpatioTemporalFusion", side_effect=AssertionError("unused fusion")):
        model = AdvSTAD_NoSpatial(3, settings("concat")).eval()
    for name in ("sensor_embedding", "spatial_input_projection", "spatial_dropout", "spatial_encoder", "fusion"):
        assert not hasattr(model, name)
    assert not any("spatial" in name or "fusion" in name or "sensor_embedding" in name
                   for name, _ in model.named_parameters())
    src = torch.rand(4, 2, 3)
    temporal_outputs = []
    hook = model.temporal_encoder[-1].register_forward_hook(
        lambda module, inputs, output: temporal_outputs.append(output)
    )
    try:
        memory = model.encode(src, torch.zeros_like(src))
    finally:
        hook.remove()
    assert memory is temporal_outputs[-1]
    assert memory.shape == (4, 2, 8)


def test_temporal_only_rejects_invalid_shapes_dtype_and_batch_alignment():
    model = AdvSTAD_NoSpatial(3, settings())
    src = torch.rand(4, 2, 3)
    for bad_src, bad_tgt in ((src[:3], src[-1:]), (src, src[-1:, :1]),
                             (src, src), (src.double(), src[-1:].double())):
        with pytest.raises(ValueError):
            model(bad_src, bad_tgt)
    with pytest.raises(ValueError, match="conditioning"):
        model.encode(src, torch.zeros(4, 1, 3))


def test_inactive_spatial_settings_do_not_change_temporal_architecture_or_rng():
    initializations = []
    for fusion, depth in zip(FUSIONS, (1, 2, 3)):
        torch.manual_seed(32)
        model = AdvSTAD_NoSpatial(3, settings(fusion, spatial_layers=depth))
        initializations.append((model, torch.get_rng_state()))
    reference, reference_rng = initializations[0]
    for model, rng in initializations:
        assert model.resolved_config["fusion"] == "temporal_only"
        assert model.resolved_config["spatial_layers"] == 0
        assert model.config is model.resolved_config
        assert model.resolved_config == reference.resolved_config
        assert experiment_key(model, "SMD") == experiment_key(reference, "SMD")
        assert torch.equal(rng, reference_rng)
        for name, value in model.state_dict().items():
            assert torch.equal(value, reference.state_dict()[name])
    rebuilt = AdvSTAD_NoSpatial(3, reference.resolved_config)
    assert rebuilt.resolved_config == reference.resolved_config


def test_no_spatial_reuses_temporal_decoder_and_optimizer_configuration():
    supplied = settings("cross_attention", temporal_layers=2, decoder_layers=2)
    original = AdvSTAD(3, supplied)
    ablated = AdvSTAD_NoSpatial(3, supplied)
    assert supplied["fusion"] == "cross_attention"
    assert len(ablated.temporal_encoder) == len(original.temporal_encoder) == 2
    for name in ("temporal_input_projection", "target_projection", "temporal_encoder", "decoder1", "decoder2", "head1", "head2"):
        actual, expected = getattr(ablated, name), getattr(original, name)
        assert type(actual) is type(expected)
        assert {key: value.shape for key, value in actual.state_dict().items()} == {
            key: value.shape for key, value in expected.state_dict().items()
        }
    groups = parameter_groups(ablated)
    generator = {id(parameter) for parameter in groups["generator"]}
    adversary = {id(parameter) for parameter in groups["adversary"]}
    assert generator.isdisjoint(adversary)
    assert generator | adversary == {id(parameter) for parameter in ablated.parameters()}
    assert adversary == {id(parameter) for module in (ablated.decoder2, ablated.head2)
                         for parameter in module.parameters()}
    optimizers, schedulers = build_optimizers(ablated)
    for owner in ("generator", "adversary"):
        assert isinstance(optimizers[owner], torch.optim.AdamW)
        assert optimizers[owner].param_groups[0]["lr"] == ablated.lr == original.lr
        assert optimizers[owner].param_groups[0]["weight_decay"] == 1e-5
        assert schedulers[owner].step_size == 5
        assert schedulers[owner].gamma == 0.9


def test_no_spatial_synthetic_training_scoring_and_checkpoint_roundtrip():
    torch.manual_seed(101)
    model = AdvSTAD_NoSpatial(3, settings())
    data = torch.rand(3, 4, 3)
    optimizers, schedulers = build_optimizers(model)
    metrics = train_epoch(model, data, optimizers, schedulers, epoch=2)
    assert all(np.isfinite(value) for value in metrics.values())
    scores, predictions = evaluate(model, data)
    assert scores.shape == predictions.shape == (3, 3)
    np.testing.assert_allclose(scores, (predictions - data[:, -1].numpy()) ** 2)
    payload = checkpoint_payload(model, optimizers, schedulers, 2, [(metrics["loss_g"], metrics["lr_g"])])
    restored = AdvSTAD_NoSpatial(3, settings("concat"))
    restored_optimizers, restored_schedulers = build_optimizers(restored)
    epoch, _ = restore_checkpoint(payload, restored, restored_optimizers, restored_schedulers)
    assert epoch == 2
    for actual, expected in zip(evaluate(restored, data), (scores, predictions)):
        np.testing.assert_array_equal(actual, expected)
    assert all(optimizer.state for optimizer in restored_optimizers.values())


@pytest.mark.parametrize("source_class,target_class", [(AdvSTAD, AdvSTAD_NoSpatial), (AdvSTAD_NoSpatial, AdvSTAD)])
def test_checkpoint_rejects_other_model_identity_before_mutating_state(source_class, target_class):
    source, target = source_class(3, settings()), target_class(3, settings())
    optimizers, schedulers = build_optimizers(source)
    payload = checkpoint_payload(source, optimizers, schedulers, -1, [])
    assert checkpoint_metadata(source)["model"] == source.name
    assert checkpoint_metadata(target)["model"] == target.name
    assert experiment_key(source, "SMD") != experiment_key(target, "SMD")
    target_state = copy.deepcopy(target.state_dict())
    target_optimizers, target_schedulers = build_optimizers(target)
    with pytest.raises(ValueError, match="metadata.model"):
        restore_checkpoint(payload, target, target_optimizers, target_schedulers)
    for name, value in target.state_dict().items():
        assert torch.equal(value, target_state[name])
    assert all(not optimizer.state for optimizer in target_optimizers.values())


@pytest.mark.parametrize("fusion,expected_count,expected_key", [
    ("sum", 3178, "AdvSTAD_SMD_sum_f92305b1641d"),
    ("concat", 3314, "AdvSTAD_SMD_concat_6165f7e78295"),
    ("cross_attention", 3398, "AdvSTAD_SMD_cross_attention_37ebbc6b94c8"),
])
def test_original_advstad_parameter_counts_and_checkpoint_fingerprints_stay_compatible(fusion, expected_count, expected_key):
    # Frozen before the ablation implementation; no training duration/runtime
    # values influence the original architecture's stable checkpoint identity.
    config = settings(fusion)
    config.pop("training")
    model = AdvSTAD(3, config)
    assert parameter_count(model) == expected_count
    assert experiment_key(model, "SMD") == expected_key
    assert model.resolved_config["fusion"] == fusion
    assert model.resolved_config["spatial_layers"] == 1
    assert checkpoint_metadata(model)["model"] == "AdvSTAD"
    assert parameter_count(AdvSTAD_NoSpatial(3, config)) == 2446


def test_four_model_parameter_counts_for_current_38_sensor_concat_configuration():
    assert parameter_count(TranAD(38)) == 127538
    assert parameter_count(TranAD_Corrected(38)) == 127538
    assert parameter_count(AdvSTAD_NoSpatial(38, {"fusion": "concat"})) == 130464
    assert parameter_count(AdvSTAD(38, {"fusion": "concat"})) == 176242
