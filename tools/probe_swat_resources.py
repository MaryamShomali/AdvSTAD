#!/usr/bin/env python3
"""Read-only, bounded SWaT checkpoint probe; no training or application artifacts.

venv/bin/python tools/probe_swat_resources.py --model TranAD \
    --checkpoint checkpoints/TranAD_SWaT/model.ckpt \
    --output runs/resource-diagnosis/checkpoint-probe.json

Samples real, consecutive windows spread over the training timeline. Reports
host/CUDA memory at startup stages and tests SPOT initialization ONCE for each
constant sampled reconstruction score. It never enters pot_eval's retry loop.
Checkpoint files must be trusted, as with the application's own torch.load.
"""

import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

from profile_resources import proc_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=("TranAD", "AdvSTAD"), default="TranAD")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--output", required=True)
    options = parser.parse_args()
    if options.samples <= 0:
        parser.error("--samples must be positive")
    output = Path(options.output).resolve()
    if output.exists():
        parser.error("output already exists: {}".format(output))
    checkpoint_path = Path(options.checkpoint).resolve()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    # Repository imports parse argv at import time. This affects only this probe.
    sys.argv = ["swat-resource-probe", "--model", options.model, "--dataset", "SWaT"]
    started = time.monotonic()
    report = {"model": options.model, "checkpoint": str(checkpoint_path), "stages": []}

    def stage(name):
        record = {"stage": name, "elapsed_seconds": time.monotonic() - started,
                  "memory": proc_memory(os.getpid()),
                  "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            record["cuda_allocated_bytes"] = torch.cuda.memory_allocated()
            record["cuda_reserved_bytes"] = torch.cuda.memory_reserved()
            record["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["stages"].append(record)
        print(json.dumps(record), flush=True)

    stage("stdlib_only")
    import numpy as np
    import torch
    stage("numpy_and_torch_imported")
    from src.config import CONFIG, DEVICE, DTYPE, get_advstad_config
    from src.constants import lr_d
    import src.models as models
    from src.spot import SPOT
    stage("repository_imported")
    metadata = json.loads((root / "processed/SWaT/metadata.json").read_text())
    values = np.load(root / "processed/SWaT/train.npy", mmap_mode="r")
    if options.model == "TranAD":
        model = models.TranAD(values.shape[1])
    else:
        settings = get_advstad_config(CONFIG, values.shape[1], lr_d["SWaT"])
        model = models.AdvSTAD(values.shape[1], settings)
    stage("model_on_cpu")
    model.to(device=DEVICE, dtype=DTYPE)
    stage("model_on_configured_device")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    stage("checkpoint_loaded")
    indices = np.linspace(0, len(values) - 1, min(options.samples, len(values)), dtype=np.int64)
    numpy_dtype = np.float32 if DTYPE == torch.float32 else np.float64
    windows = np.empty((len(indices), model.n_window, values.shape[1]), dtype=numpy_dtype)
    for k, index in enumerate(indices):
        # Match main.convert_to_windows endpoint alignment for each model.
        end = index + (1 if options.model == "AdvSTAD" else 0)
        windows[k] = values[np.maximum(np.arange(end - model.n_window, end), 0)]
    inputs = torch.from_numpy(windows).to(device=DEVICE, dtype=DTYPE)
    stage("sample_windows_on_device")
    outputs = []
    with torch.no_grad():
        for batch in inputs.split(model.batch):
            window = batch.permute(1, 0, 2)
            outputs.append(model(window, window[-1:])[1][0].cpu())
    predictions = torch.cat(outputs).numpy()
    targets = windows[:, -1].astype(predictions.dtype, copy=False)
    scores = (predictions - targets) ** 2
    stage("sample_inference_finished")
    report.update(device=str(next(model.parameters()).device),
                  dtype=str(next(model.parameters()).dtype),
                  checkpoint_epoch=checkpoint.get("epoch"),
                  sample_windows=len(indices), constant_score_features=[])
    for j, name in enumerate(metadata["feature_names"]):
        score = scores[:, j]
        if score.min() != score.max():
            continue
        spot = SPOT(1e-5)
        spot.fit(score, score)
        try:
            spot.initialize(level=.993, min_extrema=False, verbose=False)
            outcome = "ok"
        except Exception as error:
            outcome = "{}: {}".format(type(error).__name__, error)
        report["constant_score_features"].append({
            "index": j, "feature": name, "score": float(score[0]),
            "prediction_min": float(predictions[:, j].min()),
            "prediction_max": float(predictions[:, j].max()),
            "spot_initialize": outcome,
        })
    stage("spot_probes_finished")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("Report: {}".format(output), flush=True)


if __name__ == "__main__":
    main()
