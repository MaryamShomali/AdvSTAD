# AdvSTAD: Dual-Route Spatio-Temporal Transformer with Adversarial Learning for Multivariate Time Series Anomaly Detection

## Running tracked experiments

Install the dependencies, preprocess the selected dataset, and run the existing entry point. Experiment tracking is enabled in `config.yaml` by default.

```bash
pip install -r requirements.txt
python preprocess.py SMD
python main.py --model TranAD --dataset SMD --retrain --run-name gpu-float32
```

Every run creates `runs/<run-id>/run.json`, `feature_metrics.csv`, and a `tensorboard/` event directory. Completed and failed run records are also appended to `experiments.jsonl`. The JSON fields mirror the label, model, dataset, date, and result information in `experiments.md`, and add configuration, dataset, environment, hyperparameter, checkpoint, timing, and artifact metadata. A `--test` run is tracked in the same way; training metrics are simply absent. This project has no separate validation phase, so the tracker does not create or evaluate a new validation split.

View all runs with:

```bash
tensorboard --logdir runs
```

Then open the URL printed by TensorBoard (normally `http://localhost:6006`). Set `experiment_tracking.enabled` to `false` in `config.yaml` to run without creating tracking artifacts.
