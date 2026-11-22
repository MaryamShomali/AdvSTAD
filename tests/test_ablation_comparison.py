"""Synthetic saved records only: no training, inference, or dataset reads."""

import copy
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from ablation_study import (
    ABLATION_MODELS,
    SMD_MACHINE_ID,
    comparison_rows,
    configured_runs_dir,
    discover_runs,
    main,
    select_runs,
    write_csv,
)


def record(model, run_id=None, seed=1, completed="2026-11-22T10:00:00+03:30"):
    result = {
        "run_id": run_id or model + "-run", "model": model, "dataset": "SMD",
        "status": "completed", "seed": seed,
        "timestamps": {"started_at": completed, "completed_at": completed},
        "mode": {"less_data": False, "test_only": False, "retrain": True},
        "dataset_details": {
            "machine_id": SMD_MACHINE_ID, "train_shape": [12, 3],
            "test_shape": [8, 3], "labels_shape": [8, 3],
            "data_fingerprints": {"train": "train-hash", "test": "test-hash", "labels": "labels-hash"},
            "window_alignment": "legacy_previous_endpoint" if model == "TranAD" else "inclusive_endpoint_v1",
        },
        "environment": {"device": "cpu", "dtype": "float32", "pytorch": "test-version",
                        "cpu": "test-cpu", "torch_threads": 1},
        "reproducibility": {"torch_seed": seed, "numpy_seed": None, "python_seed": None,
                            "deterministic_algorithms": False, "cudnn_deterministic": False,
                            "cudnn_benchmark": False},
        "evaluation_protocol": {
            "version": "legacy_pot_point_adjusted_v1", "score": "decoder2_endpoint_squared_error",
            "aggregation": "mean_across_sensors", "labels": "any_positive_sensor",
            "point_adjustment": True, "roc_auc_input": "adjusted_binary_predictions",
            "pot": {"q": 1e-5, "level": .9999, "multiplier": 1.04, "dynamic": False, "max_attempts": 10},
        },
        "hyperparameters": {
            "starting_epoch": -1, "epochs_per_run": 5,
            "model": {"lr": .001, "batch": 128, "n_window": 10, "n_feats": 3},
            "trainable_parameter_count": 120,
            "optimizer": {"name": "AdamW", "learning_rate": .001, "weight_decay": 1e-5},
            "scheduler": {"name": "StepLR", "step_size": 5, "gamma": .9},
        },
        "checkpoint": {"path": f"checkpoints/{model}/model.ckpt", "epoch": 4},
        "git": {"revision": "same-revision", "dirty": False},
        "metrics": {
            "evaluation": {"overall": {"precision": .9, "recall": .8, "f1": .85, "ROC/AUC": .87, "TP": 7}},
            "runtime": {
                "training_total_seconds": 10., "training_average_epoch_seconds": 2.,
                "inference_total_seconds": .04, "inference_latency_ms_per_timepoint": 5.,
                "training_epochs_this_run": 5, "training_epochs_total": 5,
                "timing_version": "synchronized_perf_counter_v1",
            },
            "memory": {"cuda_peak_allocated_bytes": 2_000_000, "process_peak_rss_bytes": 30_000_000},
        },
    }
    if model.startswith("AdvSTAD"):
        hyper = result["hyperparameters"]
        hyper["optimizers"] = {name: copy.deepcopy(hyper["optimizer"]) for name in ("generator", "adversary")}
        hyper["schedulers"] = {name: copy.deepcopy(hyper["scheduler"]) for name in ("generator", "adversary")}
        del hyper["optimizer"], hyper["scheduler"]
        hyper["advstad"] = {
            "window_size": 10, "batch_size": 128, "d_model": 6, "nhead": 3,
            "temporal_layers": 1, "decoder_layers": 1, "dim_feedforward": 16, "dropout": .1,
            "spatial_layers": 0 if model.endswith("NoSpatial") else 1,
            "fusion": "temporal_only" if model.endswith("NoSpatial") else "concat",
            "training": {"learning_rate": .001, "adversarial_weight": .1,
                         "gradient_clip_norm": 1., "epochs_per_run": 5},
        }
        hyper["checkpoint_metadata"] = {
            "objective_version": "anchored_adversarial_v1", "schedule_version": "absolute_epoch_growth_v1",
        }
    return result


def save(runs_dir, data):
    destination = runs_dir / data["run_id"] / "run.json"
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps(data), encoding="utf-8")
    return destination


def compare(runs_dir, **kwargs):
    runs, warnings = discover_runs(runs_dir)
    return select_runs(runs, warnings=warnings, **kwargs)


def table(comparison):
    return {row[0]: row[1:] for row in comparison_rows(comparison)[1:]}


def test_four_models_column_order_metrics_runtime_and_csv(tmp_path):
    paths = [save(tmp_path / "runs", record(model)) for model in reversed(ABLATION_MODELS)]
    before = {path: path.read_bytes() for path in paths}
    comparison = compare(tmp_path / "runs")
    assert list(comparison_rows(comparison)[0]) == [
        "Metric", "TranAD", "TranAD_Corrected", "AdvSTAD_NoSpatial", "AdvSTAD",
    ]
    assert set(comparison.selected) == set(ABLATION_MODELS)
    rows = table(comparison)
    assert rows["Precision"] == ["0.9"] * 4
    assert rows["Total training time (seconds)"] == ["10"] * 4
    assert rows["Average training time per epoch (seconds)"] == ["2"] * 4
    assert rows["Total inference time (seconds)"] == ["0.04"] * 4
    assert rows["Average inference latency per timepoint (milliseconds)"] == ["5"] * 4
    assert rows["Number of trainable parameters"] == ["120"] * 4
    assert rows["Peak GPU memory (MB)"] == ["2"] * 4
    assert rows["Peak RAM (MB; process lifetime)"] == ["30"] * 4
    assert "Accuracy" not in rows
    destination = write_csv(comparison, tmp_path / "output")
    with destination.open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.reader(stream))
    assert csv_rows == comparison_rows(comparison)
    assert all(len(row) == 5 for row in csv_rows)
    assert {path: path.read_bytes() for path in paths} == before


def test_missing_and_incomplete_models_keep_na_columns(tmp_path):
    save(tmp_path, record("TranAD"))
    failed = record("AdvSTAD")
    failed["status"] = "failed"
    save(tmp_path, failed)
    incomplete = record("TranAD_Corrected")
    incomplete["metrics"]["evaluation"] = {}
    save(tmp_path, incomplete)
    comparison = compare(tmp_path)
    assert set(comparison.selected) == {"TranAD"}
    assert table(comparison)["F1-score"] == ["0.85", "N/A", "N/A", "N/A"]
    assert any("failed" in warning for warning in comparison.warnings)
    assert any("no completed evaluation" in warning for warning in comparison.warnings)


def test_missing_machine_is_excluded_even_when_shapes_match(tmp_path):
    unknown = record("TranAD")
    del unknown["dataset_details"]["machine_id"]
    other = record("TranAD_Corrected")
    other["dataset_details"]["machine_id"] = "machine-1-2"
    save(tmp_path, unknown)
    save(tmp_path, other)
    comparison = compare(tmp_path)
    assert not comparison.selected
    assert all(value == "N/A" for row in comparison_rows(comparison)[1:] for value in row[1:])
    assert any("shapes do not establish" in warning for warning in comparison.warnings)


@pytest.mark.parametrize("path,value", [
    (("seed",), 2), (("dataset_details", "data_fingerprints", "train"), "different"),
    (("hyperparameters", "model", "n_window"), 20),
    (("environment", "dtype"), "float64"),
    (("environment", "cpu"), "different-cpu"),
    (("environment", "torch_threads"), 4),
    (("metrics", "runtime", "training_epochs_total"), 10),
    (("mode", "less_data"), True),
    (("evaluation_protocol", "pot", "level"), .99),
    (("reproducibility", "numpy_seed"), 1),
])
def test_incompatible_controls_are_not_combined(tmp_path, path, value):
    save(tmp_path, record("TranAD"))
    different = record("TranAD_Corrected")
    nested = different
    for key in path[:-1]:
        nested = nested[key]
    nested[path[-1]] = value
    save(tmp_path, different)
    comparison = compare(tmp_path)
    assert len(comparison.selected) == 1
    assert any("incompatible controls" in warning for warning in comparison.warnings)


def test_cohort_coverage_precedes_recency_and_models_never_mix_seeds(tmp_path):
    for model in ABLATION_MODELS:
        save(tmp_path, record(model, model + "-old", completed="2026-11-20T10:00:00Z"))
    save(tmp_path, record("TranAD", "TranAD-new-seed", seed=2, completed="2026-11-22T10:00:00Z"))
    comparison = compare(tmp_path)
    assert len(comparison.selected) == 4
    assert {run.record["seed"] for run in comparison.selected.values()} == {1}


def test_latest_run_within_cohort_and_explicit_selection(tmp_path):
    old = record("TranAD", "old", completed="2026-11-21T22:00:00+03:30")
    latest = record("TranAD", "new", completed="2026-11-21T19:00:00Z")
    save(tmp_path, old)
    save(tmp_path, latest)
    save(tmp_path, record("TranAD_Corrected"))
    assert compare(tmp_path).selected["TranAD"].run_id == "new"
    chosen = compare(tmp_path, explicit={"TranAD": "old"})
    assert chosen.selected["TranAD"].run_id == "old"
    assert len(chosen.selected) == 2
    missing = compare(tmp_path, explicit={"TranAD": "does-not-exist"})
    assert "TranAD" not in missing.selected
    assert any("unavailable or ineligible" in warning for warning in missing.warnings)


def test_incompatible_explicit_pins_use_first_reference(tmp_path):
    save(tmp_path, record("TranAD", "seed-one", seed=1))
    save(tmp_path, record("TranAD_Corrected", "seed-two", seed=2))
    chosen = compare(tmp_path, explicit={"TranAD": "seed-one", "TranAD_Corrected": "seed-two"})
    assert set(chosen.selected) == {"TranAD"}
    assert any("conflicts with the first selected reference" in warning for warning in chosen.warnings)


def test_unknown_metadata_is_a_single_unverified_reference(tmp_path):
    for model in ABLATION_MODELS:
        historic = record(model)
        del historic["seed"]
        del historic["evaluation_protocol"]
        save(tmp_path, historic)
    comparison = compare(tmp_path)
    assert len(comparison.selected) == 1
    assert "Unverified reference" in table(comparison)["Comparability"]
    assert any("evaluation_protocol" in warning and "seed" in warning for warning in comparison.warnings)


def test_known_cohort_preferred_to_newer_unknown_records(tmp_path):
    save(tmp_path, record("TranAD", "verified", completed="2026-11-20T10:00:00Z"))
    unknown = record("AdvSTAD", "unknown", completed="2026-11-22T10:00:00Z")
    del unknown["dataset_details"]["data_fingerprints"]
    save(tmp_path, unknown)
    assert set(compare(tmp_path).selected) == {"TranAD"}


def test_spatial_pair_ignores_removed_path_but_checks_effective_temporal_settings(tmp_path):
    save(tmp_path, record("AdvSTAD_NoSpatial"))
    spatial = record("AdvSTAD")
    spatial["hyperparameters"]["advstad"]["fusion"] = "cross_attention"
    save(tmp_path, spatial)
    assert len(compare(tmp_path).selected) == 2
    path = tmp_path / spatial["run_id"] / "run.json"
    spatial["hyperparameters"]["advstad"]["dropout"] = .2
    path.write_text(json.dumps(spatial), encoding="utf-8")
    comparison = compare(tmp_path)
    assert len(comparison.selected) == 1
    assert any("advstad.dropout" in warning for warning in comparison.warnings)


def test_spatial_pair_prefers_older_matching_settings_to_newer_mismatch(tmp_path):
    save(tmp_path, record("AdvSTAD_NoSpatial"))
    save(tmp_path, record("AdvSTAD", "matching", completed="2026-11-21T10:00:00Z"))
    different = record("AdvSTAD", "different", completed="2026-11-22T10:00:00Z")
    different["hyperparameters"]["advstad"]["training"]["adversarial_weight"] = .5
    save(tmp_path, different)
    comparison = compare(tmp_path)
    assert len(comparison.selected) == 2
    assert comparison.selected["AdvSTAD"].run_id == "matching"


def test_test_only_runtime_stays_missing_and_not_borrowed_from_training(tmp_path):
    trained = record("TranAD", "training-run", completed="2026-11-21T10:00:00Z")
    save(tmp_path, trained)
    tested = record("TranAD", "test-run", completed="2026-11-22T10:00:00Z")
    tested["mode"]["test_only"] = True
    tested["metrics"]["runtime"].update(training_total_seconds=None,
        training_average_epoch_seconds=None, training_epochs_this_run=0)
    tested["metrics"]["evaluation"]["overall"]["accuracy"] = .91
    save(tmp_path, tested)
    comparison = compare(tmp_path)
    rows = table(comparison)
    assert rows["Run ID"][0] == "test-run"
    assert rows["Total training time (seconds)"][0] == "N/A"
    assert rows["Total inference time (seconds)"][0] == "0.04"
    assert rows["Accuracy"][0] == "0.91"
    assert any("never borrowed" in warning for warning in comparison.warnings)


def test_decayed_checkpoint_optimizer_lr_keeps_base_lr_cohort_compatible(tmp_path):
    save(tmp_path, record("TranAD"))
    tested = record("TranAD_Corrected", "checkpoint-test")
    tested["mode"]["test_only"] = True
    tested["hyperparameters"]["starting_epoch"] = 4
    tested["hyperparameters"]["optimizer"]["learning_rate"] = .0009
    tested["metrics"]["runtime"].update(training_total_seconds=None,
        training_average_epoch_seconds=None, training_epochs_this_run=0)
    save(tmp_path, tested)
    comparison = compare(tmp_path)
    assert len(comparison.selected) == 2
    assert table(comparison)["Total training time (seconds)"][:2] == ["10", "N/A"]
    assert table(comparison)["Number of training epochs (total)"][:2] == ["5", "5"]
    assert any("not cumulative checkpoint training costs" in warning for warning in comparison.warnings)


def test_legacy_evaluation_timer_is_never_relabelled_inference(tmp_path):
    old = record("TranAD")
    del old["metrics"]["runtime"]
    old["metrics"]["timing_seconds"] = {"training": 100., "evaluation": 12.}
    save(tmp_path, old)
    comparison = compare(tmp_path)
    rows = table(comparison)
    assert rows["Total inference time (seconds)"][0] == "N/A"
    assert rows["Total training time (seconds)"][0] == "N/A"
    assert any("Legacy evaluation timing" in warning for warning in comparison.warnings)


def test_malformed_json_and_missing_directories_are_graceful(tmp_path, capsys):
    broken = tmp_path / "bad" / "run.json"
    broken.parent.mkdir()
    broken.write_text("{not-json", encoding="utf-8")
    comparison = compare(tmp_path)
    assert not comparison.selected
    assert any("Cannot read" in warning for warning in comparison.warnings)
    output = tmp_path / "output"
    assert main(["--compare", "--runs-dir", str(tmp_path / "missing"), "--output-dir", str(output)]) == 0
    assert (output / "ablation_comparison.csv").is_file()
    assert "Run directory does not exist" in capsys.readouterr().out


def test_cli_seed_filter_and_configured_log_directory(tmp_path, capsys):
    (tmp_path / "config.yaml").write_text("experiment_tracking:\n  log_dir: archived-runs\n", encoding="utf-8")
    assert configured_runs_dir(tmp_path) == tmp_path / "archived-runs"
    save(tmp_path / "archived-runs", record("TranAD", "seed-one", seed=1))
    save(tmp_path / "archived-runs", record("TranAD_Corrected", "seed-two", seed=2))
    output = tmp_path / "result"
    assert main(["--compare", "--runs-dir", str(tmp_path / "archived-runs"), "--seed", "2",
                 "--run", "TranAD_Corrected=seed-two", "--output-dir", str(output)]) == 0
    with (output / "ablation_comparison.csv").open(newline="", encoding="utf-8") as stream:
        rows = {row[0]: row[1:] for row in list(csv.reader(stream))[1:]}
    assert rows["Run ID"] == ["N/A", "seed-two", "N/A", "N/A"]
    assert "seed does not match" in capsys.readouterr().out


def test_comparison_import_does_not_import_torch_models_or_main():
    project = Path(__file__).resolve().parents[1]
    command = ("import sys; import ablation_study; "
               "assert not ({'torch', 'src.models', 'src.parser', 'main'} & set(sys.modules))")
    result = subprocess.run([sys.executable, "-c", command], cwd=str(project), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("arguments", [[], ["--compare", "--run", "Unknown=run"],
    ["--compare", "--run", "TranAD="], ["--compare", "--run", "TranAD=a", "--run", "TranAD=b"]])
def test_invalid_cli_arguments_fail_without_running_pipeline(arguments):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
