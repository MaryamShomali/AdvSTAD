"""Behavioral checks for opposing updates, gradient paths, and ordered scoring."""

import copy

import numpy as np
import pytest
import torch

from src import advstad_training as training
from src.config import get_advstad_config
from src.models import AdvSTAD


def make_model(fusion="sum", dtype=torch.float64, batch_size=2, device="cpu"):
    torch.manual_seed(37)
    config = get_advstad_config(
        {
            "advstad": {
                "window_size": 4,
                "batch_size": batch_size,
                "dropout": 0.0,
                "fusion": fusion,
                "training": {"learning_rate": 0.003},
            }
        },
        feats=3,
        default_lr=0.001,
    )
    return AdvSTAD(3, config).to(device=device, dtype=dtype)


def snapshot(parameters):
    return [parameter.detach().clone() for parameter in parameters]


def assert_unchanged(parameters, previous):
    assert all(torch.equal(parameter, before) for parameter, before in zip(parameters, previous))


def assert_updated(parameters, previous):
    assert any(not torch.equal(parameter, before) for parameter, before in zip(parameters, previous))


def assert_nonzero_gradient(parameters):
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


@pytest.mark.parametrize("fusion", ["sum", "concat", "cross_attention"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_alternating_updates_respect_disjoint_parameter_ownership(fusion, dtype, monkeypatch):
    model = make_model(fusion, dtype)
    groups = training.parameter_groups(model)
    generator_ids = {id(parameter) for parameter in groups["generator"]}
    adversary_ids = {id(parameter) for parameter in groups["adversary"]}
    assert generator_ids.isdisjoint(adversary_ids)
    assert generator_ids | adversary_ids == {id(parameter) for parameter in model.parameters()}
    assert adversary_ids == {
        id(parameter)
        for module in (model.decoder2, model.head2)
        for parameter in module.parameters()
    }
    optimizers, _ = training.build_optimizers(model)
    before = {owner: snapshot(parameters) for owner, parameters in groups.items()}
    steps = []

    def step_spy(owner, inactive):
        original = optimizers[owner].step

        def step(*args, **kwargs):
            assert model.training
            assert all(parameter.requires_grad for parameter in groups[owner])
            assert all(not parameter.requires_grad for parameter in groups[inactive])
            assert all(parameter.grad is None for parameter in groups[inactive])
            assert_nonzero_gradient(groups[owner])
            frozen = snapshot(groups[inactive])
            result = original(*args, **kwargs)
            assert_unchanged(groups[inactive], frozen)
            assert_updated(groups[owner], before[owner])
            steps.append(owner)
            return result

        return step

    monkeypatch.setattr(optimizers["adversary"], "step", step_spy("adversary", "generator"))
    monkeypatch.setattr(optimizers["generator"], "step", step_spy("generator", "adversary"))
    data = torch.rand(2, 4, 3, dtype=dtype)
    metrics = training.alternating_step(model, data, optimizers, epoch=2)
    assert steps == ["adversary", "generator"]
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["alpha"] == pytest.approx(0.1 * (1 - 1 / 3))
    assert metrics["loss_g"] == pytest.approx(
        metrics["error_y1"] + metrics["alpha"] * metrics["error_y2"]
    )
    assert metrics["loss_a"] == pytest.approx(
        metrics["error_y2_base_adversary"] - metrics["alpha"] * metrics["error_y2_adversary"]
    )


@pytest.mark.parametrize("fusion", ["sum", "concat", "cross_attention"])
def test_conditioned_error_reaches_decoder1_and_both_encoders_with_frozen_adversary(fusion):
    model = make_model(fusion)
    groups = training.parameter_groups(model)
    for parameter in groups["adversary"]:
        parameter.requires_grad_(False)
    src = torch.rand(4, 2, 3, dtype=torch.float64)
    prepared = model.prepare_memories(src, src[-1:])
    y1 = prepared["y1"]
    y1.retain_grad()
    y2 = model.decode2(prepared["query"], prepared["memory1"])
    # Only the conditioned loss is differentiated, so decoder 1 can receive
    # gradients solely through C1 = (y1 - src)^2 and the second encoder pass.
    (y2 - src[-1:]).square().mean().backward()
    assert y1.grad is not None and torch.count_nonzero(y1.grad) > 0
    for module in (model.decoder1, model.head1, model.temporal_encoder, model.spatial_encoder, model.fusion):
        assert_nonzero_gradient(module.parameters())
    assert all(parameter.grad is None for parameter in groups["adversary"])


def test_conditioned_objective_has_opposite_gradients_and_independent_anchors():
    y1 = torch.tensor([[[0.2, 0.6]]], requires_grad=True)
    y2_base = torch.tensor([[[0.4, 0.7]]], requires_grad=True)
    y2 = torch.tensor([[[0.1, 0.8]]], requires_grad=True)
    target = torch.tensor([[[0.3, 0.5]]])
    values = training.objectives(y1, y2_base, y2, target, alpha=0.08)
    gradient_g = torch.autograd.grad(values["loss_g"], y2, retain_graph=True)[0]
    gradient_a = torch.autograd.grad(values["loss_a"], y2, retain_graph=True)[0]
    assert torch.count_nonzero(gradient_g) > 0
    torch.testing.assert_allclose(gradient_g, -gradient_a)
    assert torch.count_nonzero(torch.autograd.grad(values["loss_g"], y1)[0]) > 0
    assert torch.count_nonzero(torch.autograd.grad(values["loss_a"], y2_base)[0]) > 0


@pytest.mark.parametrize("stage", ["prepare_memories", "forward", "adversary_step"])
def test_failure_restores_original_flags_including_preexisting_frozen_parameter(stage, monkeypatch):
    model = make_model()
    next(model.temporal_encoder.parameters()).requires_grad_(False)
    optimizers, _ = training.build_optimizers(model)
    flags = [parameter.requires_grad for parameter in model.parameters()]

    def fail(*args, **kwargs):
        raise RuntimeError("injected training failure")

    if stage == "adversary_step":
        monkeypatch.setattr(optimizers["adversary"], "step", fail)
    else:
        monkeypatch.setattr(model, stage, fail)
    with pytest.raises(RuntimeError, match="injected training failure"):
        training.alternating_step(model, torch.rand(2, 4, 3), optimizers, epoch=1)
    assert [parameter.requires_grad for parameter in model.parameters()] == flags


def test_rejects_overlapping_optimizer_groups_before_updates():
    model = make_model()
    optimizers, _ = training.build_optimizers(model)
    before = snapshot(model.parameters())
    optimizers["adversary"].param_groups[0]["params"].append(
        optimizers["generator"].param_groups[0]["params"][0]
    )
    with pytest.raises(ValueError, match="parameter ownership"):
        training.alternating_step(model, torch.rand(2, 4, 3), optimizers, epoch=1)
    assert_unchanged(model.parameters(), before)


def test_epoch_weights_short_last_batch_and_steps_schedulers_once():
    model = make_model()
    manual = copy.deepcopy(model)
    optimizers, schedulers = training.build_optimizers(model)
    manual_optimizers, _ = training.build_optimizers(manual)
    data = torch.rand(5, 4, 3, dtype=torch.float64)
    starting_epochs = {owner: scheduler.last_epoch for owner, scheduler in schedulers.items()}
    metrics = training.train_epoch(model, data, optimizers, schedulers, epoch=5)
    expected = {}
    for offset in range(0, len(data), 2):
        batch = data[offset:offset + 2]
        result = training.alternating_step(manual, batch, manual_optimizers, epoch=5)
        for name, value in result.items():
            expected[name] = expected.get(name, 0.0) + len(batch) * value / len(data)
    for name, value in expected.items():
        assert metrics[name] == pytest.approx(value, rel=1e-9, abs=1e-12)
    assert_unchanged(model.parameters(), snapshot(manual.parameters()))
    for owner, scheduler in schedulers.items():
        assert scheduler.last_epoch == starting_epochs[owner] + 1
    assert metrics["lr_g"] == metrics["lr_a"] == model.lr


def test_absolute_epoch_schedule_and_disabled_adversary_coefficient():
    model = make_model()
    assert training.adversarial_coefficient(model, 0) == 0
    assert training.adversarial_coefficient(model, 9) == pytest.approx(0.09)
    model.config["training"]["adversarial_weight"] = 0.0
    assert training.adversarial_coefficient(model, 99) == 0
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError, match="epoch"):
            training.adversarial_coefficient(model, invalid)


@pytest.mark.parametrize("fusion", ["sum", "concat", "cross_attention"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_evaluation_short_final_batch_matches_single_samples_and_endpoint_errors(fusion, dtype):
    model = make_model(fusion, dtype)
    data = torch.rand(5, 4, 3, dtype=dtype)
    modes = []
    hook = model.register_forward_pre_hook(
        lambda module, args: modes.append((module.training, torch.is_grad_enabled(), args[0].shape[1]))
    )
    scores, predictions = training.evaluate(model, data)
    hook.remove()
    assert modes == [(False, False, 2), (False, False, 2), (False, False, 1)]
    assert scores.shape == predictions.shape == (5, 3)
    assert predictions.dtype == (np.float32 if dtype == torch.float32 else np.float64)
    np.testing.assert_allclose(scores, (predictions - data[:, -1].numpy()) ** 2, rtol=1e-6, atol=1e-8)
    model.batch = 1
    single_scores, single_predictions = training.evaluate(model, data)
    np.testing.assert_allclose(predictions, single_predictions, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(scores, single_scores, rtol=1e-5, atol=1e-7)
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("fusion", ["sum", "concat", "cross_attention"])
def test_cuda_alternating_update_and_cpu_scoring(fusion):
    model = make_model(fusion, dtype=torch.float32, device="cuda")
    optimizers, schedulers = training.build_optimizers(model)
    # The batch boundary honors the model device/dtype without a CUDA-only path.
    data = torch.rand(3, 4, 3, dtype=torch.float64)
    metrics = training.train_epoch(model, data, optimizers, schedulers, epoch=1)
    scores, predictions = training.evaluate(model, data)
    assert all(np.isfinite(value) for value in metrics.values())
    assert scores.shape == predictions.shape == (3, 3)
    assert scores.dtype == predictions.dtype == np.float32
