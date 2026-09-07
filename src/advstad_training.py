"""Alternating reconstruction-adversary training and aligned AdvSTAD scoring."""

from numbers import Integral

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


OWNERS = ("generator", "adversary")


def parameter_groups(model):
    """Partition every trainable parameter, giving decoder 2 its own optimizer."""
    adversary_ids = {
        id(parameter)
        for module in (model.decoder2, model.head2)
        for parameter in module.parameters()
    }
    groups = {owner: [] for owner in OWNERS}
    for parameter in model.parameters():
        if parameter.requires_grad:
            owner = "adversary" if id(parameter) in adversary_ids else "generator"
            groups[owner].append(parameter)
    if any(not parameters for parameters in groups.values()):
        raise ValueError("AdvSTAD requires trainable generator and adversary parameters")
    return groups


def build_optimizers(model):
    """Use disjoint AdamW optimizers and one StepLR per owner."""
    groups = parameter_groups(model)
    optimizers = {
        owner: torch.optim.AdamW(parameters, lr=model.lr, weight_decay=1e-5)
        for owner, parameters in groups.items()
    }
    schedulers = {
        owner: torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        for owner, optimizer in optimizers.items()
    }
    return optimizers, schedulers


def adversarial_coefficient(model, epoch):
    """Return the coefficient for an absolute, zero-based (possibly resumed) epoch."""
    if isinstance(epoch, bool) or not isinstance(epoch, Integral) or epoch < 0:
        raise ValueError("AdvSTAD epoch must be a nonnegative integer")
    return model.config["training"]["adversarial_weight"] * (1.0 - 1.0 / (epoch + 1))


def objectives(y1, y2_base, y2, target, alpha):
    """Unsigned reconstruction errors and the two opposing minimization losses."""
    error_y1 = F.mse_loss(y1, target)
    error_y2_base = F.mse_loss(y2_base, target)
    error_y2 = F.mse_loss(y2, target)
    return {
        "error_y1": error_y1,
        "error_y2_base": error_y2_base,
        "error_y2": error_y2,
        "loss_g": error_y1 + alpha * error_y2,
        "loss_a": error_y2_base - alpha * error_y2,
    }


def _validate_data(model, data, allow_empty=False):
    if not isinstance(data, torch.Tensor):
        raise TypeError("AdvSTAD data must be a torch.Tensor with shape [N,W,F]")
    if data.ndim != 3 or tuple(data.shape[1:]) != (model.n_window, model.n_feats):
        raise ValueError(
            f"AdvSTAD data must have shape [N,{model.n_window},{model.n_feats}]; "
            f"received {tuple(data.shape)}"
        )
    if not allow_empty and len(data) == 0:
        raise ValueError("AdvSTAD training data must contain at least one window")


def _validate_optimizers(optimizers, groups):
    if set(optimizers) != set(OWNERS):
        raise ValueError("AdvSTAD optimizers must have generator and adversary keys")
    for owner, expected in groups.items():
        actual = [p for group in optimizers[owner].param_groups for p in group["params"]]
        actual_ids = [id(parameter) for parameter in actual]
        if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != {
            id(parameter) for parameter in expected
        }:
            raise ValueError(f"AdvSTAD {owner} optimizer has incompatible parameter ownership")


def _batch_inputs(model, batch):
    parameter = next(model.parameters())
    # Conversion and permutation create no writes to the overlapping window view.
    src = batch.to(device=parameter.device, dtype=parameter.dtype).permute(1, 0, 2)
    return src, src[-1:]


def alternating_step(model, batch, optimizers, epoch):
    """Perform one adversary update followed by a fresh generator update.

    Returned unsigned errors describe the fresh generator forward pass. ``loss_a``
    is the objective actually used by the earlier adversary update, whose separate
    errors are also returned so that its signed value can be interpreted exactly.
    All original parameter flags are restored even if forward/backward/step fails.
    """
    _validate_data(model, batch)
    alpha = adversarial_coefficient(model, epoch)
    groups = parameter_groups(model)
    _validate_optimizers(optimizers, groups)
    flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    clip_norm = model.config["training"]["gradient_clip_norm"]
    src, target = _batch_inputs(model, batch)
    model.train()

    def clear_gradients():
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)

    try:
        clear_gradients()
        for parameter in groups["generator"]:
            parameter.requires_grad_(False)
        with torch.no_grad():
            prepared = model.prepare_memories(src, target)
        y2_base = model.decode2(prepared["query"].detach(), prepared["memory0"].detach())
        y2 = model.decode2(prepared["query"].detach(), prepared["memory1"].detach())
        adversary_values = objectives(prepared["y1"], y2_base, y2, target.detach(), alpha)
        adversary_values["loss_a"].backward()
        if clip_norm is not None:
            nn.utils.clip_grad_norm_(groups["adversary"], clip_norm)
        optimizers["adversary"].step()

        clear_gradients()
        for parameter in groups["generator"]:
            parameter.requires_grad_(True)
        for parameter in groups["adversary"]:
            parameter.requires_grad_(False)
        # Decoder 2 remains in autograd: its input gradient must reach both
        # encoders and decoder 1 through the attached squared conditioning.
        predictions = model(src, target, return_aux=True)
        generator_values = objectives(
            predictions["y1"], predictions["y2_base"], predictions["y2"], target, alpha
        )
        generator_values["loss_g"].backward()
        if clip_norm is not None:
            nn.utils.clip_grad_norm_(groups["generator"], clip_norm)
        optimizers["generator"].step()

        metrics = {key: value.detach().item() for key, value in generator_values.items()}
        metrics.update(
            loss_a=adversary_values["loss_a"].detach().item(),
            error_y2_base_adversary=adversary_values["error_y2_base"].detach().item(),
            error_y2_adversary=adversary_values["error_y2"].detach().item(),
            alpha=alpha,
        )
        return metrics
    finally:
        for parameter, requires_grad in flags:
            parameter.requires_grad_(requires_grad)


def train_epoch(model, data, optimizers, schedulers, epoch):
    """Train ordered windows, sample-weight metrics, and advance each scheduler once."""
    _validate_data(model, data)
    if set(schedulers) != set(OWNERS):
        raise ValueError("AdvSTAD schedulers must have generator and adversary keys")
    totals = {}
    for offset in range(0, len(data), model.batch):
        batch = data[offset:offset + model.batch]
        metrics = alternating_step(model, batch, optimizers, epoch)
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + len(batch) * value
    means = {name: total / len(data) for name, total in totals.items()}
    for scheduler in schedulers.values():
        scheduler.step()
    # Match the legacy curve interface: report the rates after the epoch step.
    means["lr_g"] = optimizers["generator"].param_groups[0]["lr"]
    means["lr_a"] = optimizers["adversary"].param_groups[0]["lr"]
    return means


@torch.no_grad()
def evaluate(model, data):
    """Return ordered, featurewise endpoint squared errors and y2 predictions."""
    _validate_data(model, data, allow_empty=True)
    model.eval()
    scores, predictions = [], []
    for offset in range(0, len(data), model.batch):
        src, target = _batch_inputs(model, data[offset:offset + model.batch])
        _, y2 = model(src, target)
        scores.append((y2 - target).square()[0].cpu().numpy())
        predictions.append(y2[0].cpu().numpy())
    if not scores:
        dtype = next(model.parameters()).detach().cpu().numpy().dtype
        return np.empty((0, model.n_feats), dtype=dtype), np.empty((0, model.n_feats), dtype=dtype)
    return np.concatenate(scores, axis=0), np.concatenate(predictions, axis=0)
