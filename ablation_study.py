"""Compare saved four-model ablation records without importing the ML pipeline.

Only ``runs/<run-id>/run.json`` records are read. Selection maximizes the number
of verified, mutually compatible model columns, breaking ties by completion
time, run ID, and path. The newest run per model in that cohort is used.
Explicit ``--run MODEL=RUN_ID`` arguments pin runs; the first valid pin is the
reference when pins conflict. No values are borrowed from other records.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.study_protocol import ABLATION_MODELS, SMD_MACHINE_ID, TIMING_VERSION


PROJECT_ROOT = Path(__file__).resolve().parent
NA = "N/A"
_MISSING = object()
_ADVSTAD_MODELS = ("AdvSTAD_NoSpatial", "AdvSTAD")


def _get(record, path, default=_MISSING):
    value = record
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass
class SavedRun:
    record: dict
    path: Path
    common: dict
    temporal: dict
    unknown: list

    @property
    def model(self):
        return self.record["model"]

    @property
    def run_id(self):
        return self.record.get("run_id") or self.path.parent.name

    @property
    def recency(self):
        completed = _get(self.record, "timestamps.completed_at", "")
        try:
            date = datetime.fromisoformat(str(completed).replace("Z", "+00:00"))
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            stamp = date.timestamp()
        except (ValueError, TypeError, OverflowError):
            stamp = float("-inf")
        return stamp, self.run_id, str(self.path)


@dataclass
class Comparison:
    selected: dict
    warnings: list
    machine_id: str = SMD_MACHINE_ID


def _metadata(record):
    """Collect controls; explicit null RNG seeds mean unseeded, not absent."""
    common, temporal, unknown = {}, {}, []

    def require(path, target=common, nullable=False):
        value = _get(record, path)
        if (value is _MISSING or (value is None and not nullable) or
                value == "" or value == [] or value == {}):
            unknown.append(path)
            return
        try:
            _canonical(value)
        except (TypeError, ValueError):
            unknown.append(path + " (invalid value)")
            return
        target[path] = value

    for path in (
        "dataset_details.machine_id", "dataset_details.train_shape",
        "dataset_details.test_shape", "dataset_details.labels_shape",
        "dataset_details.data_fingerprints.train",
        "dataset_details.data_fingerprints.test",
        "dataset_details.data_fingerprints.labels", "seed",
        "environment.device", "environment.dtype", "environment.pytorch",
        "environment.cpu", "environment.torch_threads",
        "hyperparameters.model.lr", "hyperparameters.model.batch",
        "hyperparameters.model.n_window", "hyperparameters.model.n_feats",
        "metrics.runtime.training_epochs_total", "mode.less_data",
        "reproducibility.torch_seed", "reproducibility.deterministic_algorithms",
        "reproducibility.cudnn_deterministic", "reproducibility.cudnn_benchmark",
    ):
        require(path)
    for path in ("reproducibility.numpy_seed", "reproducibility.python_seed"):
        require(path, nullable=True)
    if str(_get(record, "environment.device", "")).startswith("cuda"):
        require("environment.cuda_device")

    protocol_paths = (
        "version", "score", "aggregation", "labels", "point_adjustment",
        "roc_auc_input", "pot.q", "pot.level", "pot.multiplier", "pot.dynamic",
        "pot.max_attempts",
    )
    for key in protocol_paths:
        require("evaluation_protocol." + key)
    # Extra protocol fields can change evaluation semantics, too.
    require("evaluation_protocol")

    for singular, plural in (("optimizer", "optimizers"), ("scheduler", "schedulers")):
        details = _get(record, "hyperparameters." + singular)
        if details is _MISSING:
            named = _get(record, "hyperparameters." + plural)
            details = list(named.values()) if isinstance(named, dict) and named else []
        else:
            details = [details]
        required = (("name", "weight_decay") if singular == "optimizer"
                    else ("name", "step_size", "gamma"))
        if (not details or any(not isinstance(item, dict) or
                              any(item.get(key) is None for key in required)
                              for item in details)):
            unknown.append("hyperparameters." + singular + " settings")
        else:
            # Multiplicity differs intentionally between the model families.
            # A restored optimizer can have a decayed instantaneous rate. The
            # base model LR and scheduler are the common experimental controls.
            normalized = [{key: value for key, value in item.items()
                           if singular != "optimizer" or key != "learning_rate"}
                          for item in details]
            try:
                common[singular + "_settings"] = sorted(set(_canonical(item) for item in normalized))
            except (TypeError, ValueError):
                unknown.append("hyperparameters." + singular + " settings (invalid value)")

    alignment = _get(record, "dataset_details.window_alignment")
    expected = ("legacy_previous_endpoint" if record["model"] == "TranAD"
                else "inclusive_endpoint_v1")
    if alignment != expected:
        unknown.append("dataset_details.window_alignment (expected " + expected + ")")
    # Timing methods must match; old broad evaluation timers are never used.
    require("metrics.runtime.timing_version")
    if _get(record, "metrics.runtime.timing_version", None) not in (None, TIMING_VERSION):
        unknown.append("metrics.runtime.timing_version (unsupported)")

    if record["model"] in _ADVSTAD_MODELS:
        for key in ("window_size", "batch_size", "d_model", "nhead", "temporal_layers",
                    "decoder_layers", "dim_feedforward", "dropout",
                    "training.learning_rate", "training.adversarial_weight",
                    "training.epochs_per_run"):
            require("hyperparameters.advstad." + key, target=temporal)
        require("hyperparameters.advstad.training.gradient_clip_norm", target=temporal,
                nullable=True)
        for key in ("objective_version", "schedule_version"):
            require("hyperparameters.checkpoint_metadata." + key, target=temporal)
        # spatial_layers and fusion are intentionally inactive in NoSpatial.
    return common, temporal, unknown


def discover_runs(runs_dir, machine_id=SMD_MACHINE_ID, seed=None):
    """Read summaries only; unknown machine identities are never inferred."""
    found, warnings = [], []
    for path in sorted(Path(runs_dir).glob("*/run.json")):
        try:
            with path.open(encoding="utf-8") as stream:
                record = json.load(stream)
        except (OSError, ValueError) as error:
            warnings.append(f"Cannot read {path}: {error}")
            continue
        if not isinstance(record, dict):
            warnings.append(f"Ignoring {path}: run.json must contain an object.")
            continue
        if record.get("model") not in ABLATION_MODELS or record.get("dataset") != "SMD":
            continue
        identifier = record.get("run_id") or path.parent.name
        machine = _get(record, "dataset_details.machine_id", None)
        if not machine:
            warnings.append(f"Excluded {identifier}: selected SMD machine is unknown; "
                            "historical shapes do not establish machine identity.")
            continue
        if machine != machine_id:
            warnings.append(f"Excluded {identifier}: machine {machine!r} differs from {machine_id}.")
            continue
        if record.get("status") != "completed":
            warnings.append(f"Excluded {identifier}: status is {record.get('status', 'unknown')!r}.")
            continue
        if seed is not None and record.get("seed") != seed:
            warnings.append(f"Excluded {identifier}: seed does not match --seed {seed}.")
            continue
        overall = _get(record, "metrics.evaluation.overall", {})
        if not isinstance(overall, dict) or not overall:
            warnings.append(f"Excluded {identifier}: no completed evaluation metrics.")
            continue
        common, temporal, unknown = _metadata(record)
        saved = SavedRun(record, path, common, temporal, unknown)
        if not record.get("run_id"):
            unknown.append("run_id")
        if not _get(record, "timestamps.completed_at", None):
            unknown.append("timestamps.completed_at")
        elif saved.recency[0] == float("-inf"):
            unknown.append("timestamps.completed_at (invalid timestamp)")
        found.append(saved)
        if unknown:
            warnings.append(f"Unverified metadata for {identifier}: {', '.join(unknown)}.")
    return found, warnings


def _differences(left, right):
    differences = [key for key in sorted(set(left.common) | set(right.common))
                   if left.common.get(key, _MISSING) != right.common.get(key, _MISSING)]
    if left.model in _ADVSTAD_MODELS and right.model in _ADVSTAD_MODELS:
        differences += [key for key in sorted(set(left.temporal) | set(right.temporal))
                        if left.temporal.get(key, _MISSING) != right.temporal.get(key, _MISSING)]
    return differences


def _rank(selected):
    return len(selected), tuple(sorted((run.recency for run in selected.values()), reverse=True))


def _cohort_options(group, anchor=None):
    by_model = {model: [run for run in group if run.model == model] for model in ABLATION_MODELS}
    baseline = {model: max(by_model[model], key=lambda run: run.recency)
                for model in ABLATION_MODELS[:2] if by_model[model]}
    left, right = (by_model[model] for model in _ADVSTAD_MODELS)
    if anchor is not None and anchor.model in _ADVSTAD_MODELS:
        left = [run for run in left if run.temporal == anchor.temporal]
        right = [run for run in right if run.temporal == anchor.temporal]
    pairs = []
    for first in left:
        matching = [second for second in right if second.temporal == first.temporal]
        if matching:
            second = max(matching, key=lambda run: run.recency)
            pairs.append({first.model: first, second.model: second})
    if pairs:
        for pair in pairs:
            yield {**baseline, **pair}
    else:
        for run in left + right:
            yield {**baseline, run.model: run}
        if not left and not right:
            yield baseline


def select_runs(runs, explicit=None, warnings=None, machine_id=SMD_MACHINE_ID):
    """Select a deterministic compatible cohort or one unverified reference."""
    explicit = explicit or {}
    warnings = list(warnings or [])
    available = list(runs)
    pinned = []
    for model, run_id in explicit.items():
        matches = [run for run in available if run.model == model and run.run_id == run_id]
        available = [run for run in available if run.model != model]
        if len(matches) == 1:
            available += matches
            pinned += matches
        elif matches:
            warnings.append(f"Explicit selection {model}={run_id} is ambiguous; column left N/A.")
        else:
            warnings.append(f"Explicit selection {model}={run_id} is unavailable or ineligible; column left N/A.")
    anchor = pinned[0] if pinned else None
    verified = [run for run in available if not run.unknown]
    if anchor is not None and anchor.unknown:
        selected = {anchor.model: anchor}
        warnings.append(f"{anchor.run_id} is an unverified reference only; no other model is "
                        "presented as comparable to it.")
    else:
        groups = {}
        for run in verified:
            if anchor is not None and run.common != anchor.common:
                continue
            groups.setdefault(_canonical(run.common), []).append(run)
        options = [option for group in groups.values() for option in _cohort_options(group, anchor)
                   if anchor is None or option.get(anchor.model) is anchor]
        selected = max(options, key=_rank) if options else {}
        if not selected and available:
            reference = anchor or max(available, key=lambda run: run.recency)
            selected = {reference.model: reference}
            warnings.append(f"{reference.run_id} is an unverified reference only; no other model "
                            "is presented as comparable to it.")

    reference = anchor or (max(selected.values(), key=lambda run: run.recency) if selected else None)
    for model in ABLATION_MODELS:
        if model in selected:
            continue
        candidates = [run for run in available if run.model == model]
        if candidates and reference:
            candidate = max(candidates, key=lambda run: run.recency)
            control = reference
            if model in _ADVSTAD_MODELS:
                other = _ADVSTAD_MODELS[1 - _ADVSTAD_MODELS.index(model)]
                control = selected.get(other, reference)
            reasons = _differences(candidate, control)
            if reasons:
                warnings.append(f"Excluded {candidate.run_id}: incompatible controls: {', '.join(reasons)}.")
            elif candidate.unknown or reference.unknown:
                warnings.append(f"Excluded {candidate.run_id}: comparability cannot be verified.")
        warnings.append(f"No selected completed compatible result for {model}; column contains N/A.")
    for model, run_id in explicit.items():
        if model not in selected and any(run.model == model for run in pinned):
            warnings.append(f"Explicit run {model}={run_id} conflicts with the first selected reference; "
                            "its values were not combined.")
    _add_limitations(selected, warnings)
    return Comparison(selected, list(dict.fromkeys(warnings)), machine_id)


def _add_limitations(selected, warnings):
    if not selected:
        return
    warnings.append("The timestamp alignment and independent spatial path are intended study differences. "
                    "TranAD and AdvSTAD use different objectives and optimizer update counts; equal epochs "
                    "do not imply equal computation.")
    records = [run.record for run in selected.values()]
    starts = {_canonical(_get(record, "hyperparameters.starting_epoch", None)) for record in records}
    this_run = {_canonical(_get(record, "metrics.runtime.training_epochs_this_run", None)) for record in records}
    if len(starts) > 1 or len(this_run) > 1:
        warnings.append("Training histories or epochs measured in this invocation differ. Training times "
                        "cover this run only; they are not cumulative checkpoint training costs.")
    if any(_get(record, "mode.test_only", False) for record in records):
        warnings.append("Test-only records have no measured training time; training values are never "
                        "borrowed from another run or checkpoint.")
    if any(_get(record, "metrics.memory", None) for record in records):
        warnings.append("Memory values are process-lifetime peaks, not isolated model costs. GPU memory "
                        "uses peak allocated bytes; MB means 1,000,000 bytes.")
    revisions = {_canonical(record.get("git")) for record in records}
    if len(revisions) > 1 or any(not record.get("git") or record["git"].get("dirty") for record in records):
        warnings.append("Source revision metadata differs, is missing, or contains uncommitted changes; "
                        "review source provenance before drawing causal conclusions.")
    if any(_get(record, "reproducibility.numpy_seed", _MISSING) is None or
           _get(record, "reproducibility.python_seed", _MISSING) is None or
           _get(record, "reproducibility.deterministic_algorithms", False) is False for record in records):
        warnings.append("Matching recorded seeds do not guarantee bitwise reproducibility: Python/NumPy "
                        "may be unseeded and deterministic algorithms may be disabled.")
    if any(_get(record, "evaluation_protocol.version", None) == "legacy_pot_point_adjusted_v1"
           for record in records):
        warnings.append("Metrics retain the recorded evaluation protocol: point adjustment uses test labels, "
                        "and legacy ROC/AUC is based on adjusted binary predictions rather than continuous scores.")
    for run in selected.values():
        if _get(run.record, "metrics.runtime.timing_version", None) != TIMING_VERSION:
            warnings.append(f"{run.run_id}: unsupported or absent timing version; runtime rows are N/A. "
                            "Legacy evaluation timing is not substituted for inference timing.")
        for key in ("precision", "recall", "f1"):
            if _get(run.record, "metrics.evaluation.overall." + key, None) is None:
                warnings.append(f"{run.run_id}: missing {key}; reported as N/A.")


def _format(value):
    if value is _MISSING or value is None:
        return NA
    if isinstance(value, float):
        return format(value, ".12g") if math.isfinite(value) else NA
    if isinstance(value, (dict, list)):
        return _canonical(value)
    return str(value)


def comparison_rows(comparison):
    """Return the exact five-column table used by the terminal and CSV."""
    selected = comparison.selected

    def row(label, getter):
        return [label] + [_format(getter(selected[model])) if model in selected else NA
                          for model in ABLATION_MODELS]

    def field(label, path):
        return row(label, lambda run: _get(run.record, path, None))

    def runtime(label, key):
        return row(label, lambda run: _get(run.record, "metrics.runtime." + key, None)
                   if _get(run.record, "metrics.runtime.timing_version", None) == TIMING_VERSION else None)

    rows = [
        ["Metric", *ABLATION_MODELS],
        field("Dataset", "dataset"), field("SMD machine", "dataset_details.machine_id"),
        row("Run ID", lambda run: run.run_id), field("Random seed", "seed"),
        row("Comparability", lambda run: "Unverified reference" if run.unknown else
            ("Verified cohort" if len(selected) > 1 else "Single verified reference")),
        field("Window alignment", "dataset_details.window_alignment"),
        field("Window length", "hyperparameters.model.n_window"),
        field("Checkpoint", "checkpoint.path"), field("Execution status", "status"),
        field("Precision", "metrics.evaluation.overall.precision"),
        field("Recall", "metrics.evaluation.overall.recall"),
        field("F1-score", "metrics.evaluation.overall.f1"),
    ]
    metric_keys = set()
    for run in selected.values():
        metric_keys.update(_get(run.record, "metrics.evaluation.overall", {}))
    for key in sorted(metric_keys - {"precision", "recall", "f1"}):
        label = "Accuracy" if key.lower() == "accuracy" else key
        rows.append(field(label, "metrics.evaluation.overall." + key))
    rows.extend([
        runtime("Total training time (seconds)", "training_total_seconds"),
        runtime("Average training time per epoch (seconds)", "training_average_epoch_seconds"),
        runtime("Total inference time (seconds)", "inference_total_seconds"),
        runtime("Average inference latency per timepoint (milliseconds)", "inference_latency_ms_per_timepoint"),
        field("Number of training epochs (total)", "metrics.runtime.training_epochs_total"),
        field("Number of training epochs (this run)", "metrics.runtime.training_epochs_this_run"),
        field("Number of trainable parameters", "hyperparameters.trainable_parameter_count"),
        row("Peak GPU memory (MB)", lambda run: _memory_mb(run, "cuda_peak_allocated_bytes")),
        row("Peak RAM (MB; process lifetime)", lambda run: _memory_mb(run, "process_peak_rss_bytes")),
        field("Timing version", "metrics.runtime.timing_version"),
        field("Device", "environment.device"), field("Dtype", "environment.dtype"),
    ])
    return rows


def _memory_mb(run, name):
    value = _get(run.record, "metrics.memory." + name, None)
    return value / 1_000_000 if isinstance(value, (int, float)) else None


def write_csv(comparison, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "ablation_comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(comparison_rows(comparison))
    return path


def print_table(comparison):
    rows = comparison_rows(comparison)
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    for index, row in enumerate(rows):
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))
        if index == 0:
            print("-+-".join("-" * width for width in widths))


def configured_runs_dir(project_root=PROJECT_ROOT):
    config_path = Path(project_root) / "config.yaml"
    if config_path.exists():
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        if not isinstance(config, dict):
            raise ValueError("config.yaml must contain a mapping")
        tracking = config.get("experiment_tracking") or {}
        if not isinstance(tracking, dict):
            raise ValueError("experiment_tracking must contain a mapping")
        path = Path(tracking.get("log_dir", "runs"))
    else:
        path = Path("runs")
    return path if path.is_absolute() else Path(project_root) / path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", action="store_true", help="compare saved run.json records only")
    parser.add_argument("--runs-dir", type=Path, help="override the configured run directory")
    parser.add_argument("--output-dir", type=Path, help="write the CSV into this directory")
    parser.add_argument("--run", action="append", default=[], metavar="MODEL=RUN_ID",
                        help="pin a model's saved run (repeat for more models)")
    parser.add_argument("--seed", type=int, help="consider only runs with this recorded seed")
    args = parser.parse_args(argv)
    if not args.compare:
        parser.error("--compare is required; this script only reads saved results")
    explicit = {}
    for selection in args.run:
        model, separator, run_id = selection.partition("=")
        if not separator or model not in ABLATION_MODELS or not run_id:
            parser.error("--run must be MODEL=RUN_ID using one of: " + ", ".join(ABLATION_MODELS))
        if model in explicit:
            parser.error(f"--run specified more than once for {model}")
        explicit[model] = run_id
    try:
        runs_dir = args.runs_dir if args.runs_dir is not None else configured_runs_dir()
        runs, warnings = discover_runs(runs_dir, seed=args.seed)
        if not Path(runs_dir).is_dir():
            warnings.append(f"Run directory does not exist: {runs_dir}")
        comparison = select_runs(runs, explicit, warnings)
        output_dir = args.output_dir or PROJECT_ROOT / "ablation_results" / "SMD" / SMD_MACHINE_ID
        print(f"SMD machine: {SMD_MACHINE_ID}; saved records: {runs_dir}")
        print("Selection: maximum verified model coverage, then newest completion/run ID/path; "
              "newest run per model. Explicit pins take precedence.")
        print_table(comparison)
        destination = write_csv(comparison, output_dir)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
        parser.error(str(error))
    print(f"\nCSV: {destination}")
    for warning in comparison.warnings:
        print(f"Warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
