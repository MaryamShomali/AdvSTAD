from unittest.mock import patch

import pytest
import torch
from torch import nn

from src.models import AdvSTAD, SpatioTemporalFusion


MODES = ["sum", "concat", "cross_attention"]


def make_model(mode="sum", feats=3, window=5, dtype=torch.float32, **settings):
    config = {"fusion": mode, "window_size": window, "dropout": 0.0, **settings}
    return AdvSTAD(feats, config).to(dtype=dtype)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("window,batch,feats,width,heads", [
    (10, 4, 7, 14, 7), (1, 1, 3, 6, 3), (4, 2, 1, 2, 1),
    (1, 1, 1, 2, 1), (5, 3, 3, 8, 2), (2, 1, 3, 5, 1),
])
def test_shapes_and_backward(mode, dtype, window, batch, feats, width, heads):
    torch.manual_seed(9)
    model = make_model(mode, feats, window, dtype, d_model=width, nhead=heads)
    src = torch.rand(window, batch, feats, dtype=dtype)
    prepared = model.prepare_memories(src, src[-1:])
    assert prepared["query"].shape == (1, batch, width)
    assert prepared["memory0"].shape == prepared["memory1"].shape == (window, batch, width)
    predictions = model(src, src[-1:], return_aux=True)
    assert set(predictions) == {"y1", "y2", "y2_base"}
    for output in predictions.values():
        assert output.shape == (1, batch, feats)
        assert output.dtype == dtype
        assert torch.isfinite(output).all()
        assert ((output >= 0) & (output <= 1)).all()
    sum((value - src[-1:]).square().mean() for value in predictions.values()).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.pos_encoder.pe.dtype == dtype


@pytest.mark.parametrize("mode", MODES)
def test_fusion_formula_and_selected_parameters(mode):
    fusion = SpatioTemporalFusion(mode, 3, 5, 8, 2, dropout=0).double()
    temporal = torch.randn(5, 2, 8, dtype=torch.float64)
    spatial = torch.randn(3, 2, 8, dtype=torch.float64)
    if mode == "cross_attention":
        assert not hasattr(fusion, "spatial_to_time")
        assert not hasattr(fusion, "sensor_projection")
        with patch.object(fusion.cross_attention, "forward", wraps=fusion.cross_attention.forward) as spy:
            actual = fusion(temporal, spatial)
        arguments = spy.call_args.kwargs
        assert arguments["query"] is temporal
        assert arguments["key"] is arguments["value"] is spatial
        assert arguments["need_weights"] is False
        attention, weights = fusion.cross_attention(temporal, spatial, spatial)
        assert weights.shape == (2, 5, 3)  # batch, time queries, sensor keys
        assert torch.allclose(weights.sum(-1), torch.ones(2, 5, dtype=torch.float64))
        expected = temporal + attention
    else:
        assert not hasattr(fusion, "cross_attention")
        aligned = fusion.sensor_projection(fusion.spatial_to_time(spatial).permute(2, 1, 0))
        if mode == "sum":
            assert not hasattr(fusion, "concat_projection")
            expected = temporal + aligned
        else:
            expected = fusion.concat_projection(torch.cat((temporal, aligned), dim=-1))
        actual = fusion(temporal, spatial)
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_samples_are_isolated_and_final_batch_matches_individual_inference(mode, dtype):
    torch.manual_seed(3)
    model = make_model(mode, dtype=dtype).eval()
    src = torch.rand(5, 4, 3, dtype=dtype)
    with torch.no_grad():
        batched = model(src, src[-1:])[1]
        individual = torch.cat([
            model(src[:, i:i + 1], src[-1:, i:i + 1])[1] for i in range(4)
        ], dim=1)
        changed = src.clone()
        changed[:, 1:] += 5
        perturbed = model(changed, changed[-1:])[1]
    assert torch.allclose(batched, individual, atol=1e-6, rtol=1e-5)
    assert torch.allclose(batched[:, :1], perturbed[:, :1], atol=1e-6, rtol=1e-5)


def test_attention_axes_and_exact_conditioning_for_both_routes():
    torch.manual_seed(12)
    model = make_model().double().eval()
    src = torch.rand(5, 2, 3, dtype=torch.float64)
    temporal_inputs, spatial_inputs, attention_shapes = [], [], []
    hooks = [
        model.temporal_input_projection.register_forward_pre_hook(
            lambda module, inputs: temporal_inputs.append(inputs[0].detach().clone())
        ),
        model.spatial_input_projection.register_forward_pre_hook(
            lambda module, inputs: spatial_inputs.append(inputs[0].detach().clone())
        ),
        model.temporal_encoder[0].self_attn.register_forward_pre_hook(
            lambda module, inputs: attention_shapes.append(tuple(inputs[0].shape))
        ),
        model.spatial_encoder[0].self_attn.register_forward_pre_hook(
            lambda module, inputs: attention_shapes.append(tuple(inputs[0].shape))
        ),
    ]
    try:
        y1, _ = model(src, src[-1:])
    finally:
        for hook in hooks:
            hook.remove()
    residual = (y1 - src).square()
    assert attention_shapes == [(5, 2, 6), (3, 2, 6)] * 2
    assert torch.equal(temporal_inputs[0], torch.cat((src, torch.zeros_like(src)), dim=-1))
    assert torch.equal(temporal_inputs[1], torch.cat((src, residual), dim=-1))
    assert torch.equal(spatial_inputs[0], torch.cat((src.permute(2, 1, 0), torch.zeros_like(src).permute(2, 1, 0)), dim=-1))
    assert torch.equal(spatial_inputs[1], torch.cat((src.permute(2, 1, 0), residual.permute(2, 1, 0)), dim=-1))


def test_spatial_attention_can_transfer_a_sensor_perturbation():
    torch.manual_seed(8)
    model = make_model().double().eval()
    src = torch.rand(5, 1, 3, dtype=torch.float64)
    changed = src.clone()
    changed[:, :, 0] += 0.5
    encoded = []
    hook = model.spatial_encoder[0].register_forward_hook(
        lambda module, inputs, output: encoded.append(output.detach().clone())
    )
    try:
        model.encode(src, torch.zeros_like(src))
        model.encode(changed, torch.zeros_like(changed))
    finally:
        hook.remove()
    assert not torch.allclose(encoded[0][1], encoded[1][1], atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("mode", MODES)
def test_conditioned_loss_reaches_decoder1_and_both_encoders(mode):
    torch.manual_seed(6)
    model = make_model(mode).double()
    for module in (model.decoder2, model.head2):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    src = torch.rand(5, 2, 3, dtype=torch.float64)
    prepared = model.prepare_memories(src, src[-1:])
    prepared["y1"].retain_grad()
    y2 = model.decode2(prepared["query"], prepared["memory1"])
    (y2 - src[-1:]).square().mean().backward()
    assert prepared["y1"].grad.abs().sum() > 0
    for module in (model.decoder1, model.head1, model.temporal_encoder, model.spatial_encoder):
        assert sum(parameter.grad.abs().sum() for parameter in module.parameters()) > 0
    assert all(parameter.grad is None for parameter in model.decoder2.parameters())


def test_stacks_and_heads_are_independent():
    model = make_model(temporal_layers=2, spatial_layers=2, decoder_layers=2)
    assert isinstance(model.temporal_encoder, nn.ModuleList)
    assert isinstance(model.spatial_encoder, nn.ModuleList)
    modules = [*model.temporal_encoder, *model.spatial_encoder, *model.decoder1, *model.decoder2]
    weights = [module.self_attn.in_proj_weight for module in modules]
    assert len({weight.data_ptr() for weight in weights}) == len(weights)
    assert all(not torch.equal(weights[0], weight) for weight in weights[1:])
    assert model.head1[0].weight.data_ptr() != model.head2[0].weight.data_ptr()
    assert isinstance(model.temporal_input_projection, nn.Identity)
    assert isinstance(model.target_projection, nn.Identity)


@pytest.mark.parametrize("temporal,spatial", [
    (torch.zeros(5, 2), torch.zeros(3, 2, 6)),
    (torch.zeros(4, 2, 6), torch.zeros(3, 2, 6)),
    (torch.zeros(5, 2, 6), torch.zeros(4, 2, 6)),
    (torch.zeros(5, 2, 7), torch.zeros(3, 2, 6)),
    (torch.zeros(5, 2, 6), torch.zeros(3, 1, 6)),
    (torch.zeros(5, 0, 6), torch.zeros(3, 0, 6)),
    (torch.zeros(5, 2, 6), torch.zeros(3, 2, 6, dtype=torch.float64)),
])
def test_fusion_rejects_contract_violations(temporal, spatial):
    with pytest.raises(ValueError):
        SpatioTemporalFusion("sum", 3, 5, 6, 3)(temporal, spatial)


def test_model_rejects_invalid_inputs():
    model = make_model()
    src = torch.zeros(5, 2, 3)
    for bad_src, bad_tgt in [(src[:4], src[-1:]), (src, src[-1:, :1]),
                             (src, src), (src.double(), src[-1:].double())]:
        with pytest.raises(ValueError):
            model(bad_src, bad_tgt)
    with pytest.raises(ValueError, match="conditioning"):
        model.encode(src, torch.zeros(5, 1, 3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("mode", MODES)
def test_cuda_forward_backward(mode):
    model = make_model(mode).cuda()
    src = torch.rand(5, 2, 3, device="cuda")
    outputs = model(src, src[-1:], return_aux=True)
    sum(value.square().mean() for value in outputs.values()).backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
