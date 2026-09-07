import copy

import pytest

from src.config import ADVSTAD_DEFAULTS, get_advstad_config


def test_defaults_are_resolved_without_mutating_configuration():
    supplied = {"device": "cpu", "experiment_tracking": {"enabled": False}}
    before = copy.deepcopy(supplied)
    resolved = get_advstad_config(supplied, feats=7, default_lr=0.002)
    assert resolved["d_model"] == 14
    assert resolved["nhead"] == 7
    assert resolved["window_size"] == 10
    assert resolved["training"]["learning_rate"] == 0.002
    assert resolved["training"]["epochs_per_run"] == 5
    resolved["training"]["adversarial_weight"] = 0
    assert ADVSTAD_DEFAULTS["training"]["adversarial_weight"] == 0.1
    assert supplied == before


def test_overrides_and_zero_weight_ablation():
    supplied = {"advstad": {
        "d_model": 8, "nhead": 2, "fusion": "cross_attention",
        "training": {"learning_rate": 0.01, "adversarial_weight": 0,
                     "gradient_clip_norm": None},
    }}
    before = copy.deepcopy(supplied)
    result = get_advstad_config(supplied, 3, 0.002)
    assert result["training"] == {
        "learning_rate": 0.01, "adversarial_weight": 0,
        "gradient_clip_norm": None, "epochs_per_run": 5,
    }
    assert result["d_model"] == 8
    assert result["spatial_layers"] == 1
    assert supplied == before


@pytest.mark.parametrize("config", [[], None, {"advstad": None}, {"advstad": []},
    {"advstad": {"training": None}}, {"advstad": {"training": "bad"}}])
def test_mapping_errors(config):
    with pytest.raises(ValueError, match="mapping"):
        get_advstad_config(config, 3, 0.001)


@pytest.mark.parametrize("settings", [
    {"fusion": "SUM"}, {"fusion": "unknown"}, {"fusion": None},
    {"fusoin": "sum"}, {"training": {"epoch": 2}},
    {"d_model": 7, "nhead": 3},
    {"dropout": -0.1}, {"dropout": 1}, {"dropout": float("nan")},
    {"dropout": float("inf")}, {"dropout": True},
    {"training": {"learning_rate": 0}},
    {"training": {"learning_rate": float("inf")}},
    {"training": {"adversarial_weight": -0.1}},
    {"training": {"adversarial_weight": 1.1}},
    {"training": {"adversarial_weight": float("nan")}},
    {"training": {"adversarial_weight": True}},
    {"training": {"gradient_clip_norm": 0}},
    {"training": {"gradient_clip_norm": float("inf")}},
])
def test_invalid_settings_fail_before_training(settings):
    with pytest.raises(ValueError):
        get_advstad_config({"advstad": settings}, 3, 0.001)


@pytest.mark.parametrize("key", ["window_size", "batch_size", "d_model", "nhead",
    "temporal_layers", "spatial_layers", "decoder_layers", "dim_feedforward",
    "epochs_per_run"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_positive_integers_exclude_booleans(key, value):
    settings = {key: value} if key != "epochs_per_run" else {"training": {key: value}}
    with pytest.raises(ValueError, match="positive integer"):
        get_advstad_config({"advstad": settings}, 3, 0.001)


@pytest.mark.parametrize("feats", [0, False, 1.5])
def test_feature_count_validation(feats):
    with pytest.raises(ValueError, match="feature count"):
        get_advstad_config({}, feats, 0.001)
