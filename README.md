# AdvSTAD: Dual-Route Spatio-Temporal Transformer with Adversarial Learning for Multivariate Time Series Anomaly Detection

AdvSTAD extends the repository's TranAD baseline with independent temporal and sensor attention encoders, configurable fusion, and two reconstruction decoders. Select it with `--model AdvSTAD`; existing models remain available.

```bash
python main.py --model AdvSTAD --dataset SMD --retrain --run-name sum-baseline
python main.py --model AdvSTAD --dataset SMD          # resume the matching checkpoint
python main.py --model AdvSTAD --dataset SMD --test   # evaluate the matching checkpoint
```

Configure the architecture in `config.yaml`:

```yaml
advstad:
  window_size: 10
  batch_size: 128
  d_model: null        # resolves to 2 * sensor count F
  nhead: null          # resolves to F; d_model must be divisible by nhead
  temporal_layers: 1
  spatial_layers: 1
  decoder_layers: 1
  dim_feedforward: 16
  dropout: 0.1
  fusion: sum         # sum | concat | cross_attention
  training:
    epochs_per_run: 5
    learning_rate: null     # dataset-specific baseline learning rate
    adversarial_weight: 0.1 # zero disables the opposing term
    gradient_clip_norm: 1.0 # null disables clipping
```

Feature count comes from the processed data. Window length and sensor count can differ. Both routes project into the common `d_model` width; sum and concat learn an adapter from sensor tokens to time tokens, with concat then projecting the combined embeddings back to that width. Cross-attention uses time tokens as queries and sensor tokens as keys and values, and adds the result to the temporal representation. Each mode constructs only its own fusion parameters. AdvSTAD calls the repository's custom attention blocks directly, preserving their residual connections and LeakyReLU behavior without relying on stock Transformer wrapper signatures.

For a window `X` shaped `[W,B,F]`, both encoders first receive zero conditioning. Decoder 1 reconstructs the endpoint as `y1: [1,B,F]`; the broadcast residual `C = (y1 - X)^2` then conditions a second pass through both encoders and fusion. Decoder 2 returns `y2: [1,B,F]`. The endpoint is included in the window and supplied as the decoder query, so this is reconstruction anomaly detection. Attention sees the complete observed window without causal or padding masks. Sensor identities are learned by column position; keep feature order consistent across training and inference.

With mean reconstruction error `ell`, target `Y = X[-1:]`, absolute epoch `n = epoch + 1`, and `alpha = adversarial_weight * (1 - 1/n)`, the two minimized objectives are:

```text
generator: L_G = ell(y1, Y) + alpha * ell(y2, Y)
adversary: L_A = ell(y2_base, Y) - alpha * ell(y2, Y)
```

`y2_base` reuses decoder 2 on zero-conditioned memory. Each minibatch updates decoder 2 and its head first, then updates the encoders, fusion, query projection, decoder 1 and its head. The two optimizers own disjoint parameters. During the generator update, gradients propagate through frozen decoder 2 and the conditioning residual into decoder 1. The first epoch has `alpha=0`; resumed runs continue the absolute epoch schedule. Both optimizers use AdamW with weight decay `1e-5`, and separate StepLR schedulers decay learning rates by `0.9` every five completed epochs. This opposing objective is an AdvSTAD extension; MAML is outside this implementation.

AdvSTAD uses one inclusive window per timestamp, repeating the first observation to supply initial context. With `W=3`, the first windows are `[x0,x0,x0]`, `[x0,x0,x1]`, and `[x0,x1,x2]`. Training, test scoring, training-score calibration, labels, and plotted observations share these endpoints. The legacy TranAD window convention is preserved and ends one observation earlier; account for that alignment difference in comparisons.

Evaluation returns `y2` and per-sensor scores `(y2 - Y)^2`, both shaped `[N,F]`. The same scoring procedure supplies training scores to POT. Overall scores retain the existing mean across sensors; labels are used in metrics only. Both output heads retain sigmoid, while observations are left unclipped, so inspect processed-data ranges when interpreting errors. The existing POT and diagnosis metric conventions remain in effect. In particular, signed `L_A` is a training objective, not an anomaly score or a model-selection metric. A smoke run verifies execution and makes no detection-quality claim; compare fusion modes with the same data, seed, dimensions, window policy, duration, and score definition, and report parameter count and computational cost alongside metrics.

AdvSTAD checkpoints use `checkpoints/AdvSTAD_<dataset>_<fusion>_<fingerprint>/model.ckpt`. Training/evaluation plots and tracker artifact references use the same experiment key. The stable fingerprint covers resolved architecture and training semantics, sensor count, window length, alignment, and objective/schedule versions. Device, dtype, run label, and epochs per invocation do not change the key. Consequently, changing only `--run-name` does not start a fresh model; use `--retrain` for that. Fusion or other semantic changes select a separate checkpoint. Checkpoints include both optimizer/scheduler states, completed epoch, training curves and detailed history, resolved settings, and optional sensor identifiers. Metadata is checked before loading weights, including head count and fusion, and optimizer states migrate to the selected device/dtype. Incompatible checkpoints fail with an explanation; TranAD weight conversion is not supported. Existing models keep their checkpoint formats and paths.

## Running tracked experiments

Install the dependencies, preprocess the selected dataset, and run the existing entry point. Experiment tracking is enabled in `config.yaml` by default.

```bash
pip install -r requirements.txt
python preprocess.py SMD
python main.py --model TranAD --dataset SMD --retrain --run-name gpu-float32
```

Every run creates `runs/<run-id>/run.json`, `feature_metrics.csv`, and a `tensorboard/` event directory. Completed and failed run records are also appended to `experiments.jsonl`. The JSON fields mirror the label, model, dataset, date, and result information in `experiments.md`, and add configuration, dataset, environment, hyperparameter, checkpoint, timing, and artifact metadata. A `--test` run is tracked in the same way; training metrics are simply absent. This project has no separate validation phase, so the tracker does not create or evaluate a new validation split.

AdvSTAD records resolved settings, the experiment key, both optimizers and schedulers, both learning rates, the three unsigned reconstruction errors, `L_G`, `L_A`, and `alpha`. Epoch metrics are weighted by minibatch sample count, including the final partial batch. Its existing training-curve output is labeled as the generator objective.

View all runs with:

```bash
tensorboard --logdir runs
```

Then open the URL printed by TensorBoard (normally `http://localhost:6006`). Set `experiment_tracking.enabled` to `false` in `config.yaml` to run without creating tracking artifacts.

Install the test runner with `python -m pip install pytest`, then run `python -m pytest tests`. These focused checks exercise fusion and model shapes, conditioning and gradient ownership, timestamp alignment, configuration validation, checkpoint compatibility, and small CPU updates; CUDA checks run when available.
