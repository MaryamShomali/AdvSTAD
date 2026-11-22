# Four-model ablation study

The runner supports four independent model selections. This change does not run
SMD experiments, replace either original model, or alter preprocessing, POT,
metric definitions, optimization settings, or the selected machine.

| Model | Window endpoint for score row `i` | Independent spatial route | Training |
| --- | --- | --- | --- |
| `TranAD` | `x[max(0,i-1)]`, legacy convention | No | Existing joint epoch-weighted reconstruction loss |
| `TranAD_Corrected` | `x[i]`, inclusive | No | Identical to TranAD |
| `AdvSTAD_NoSpatial` | `x[i]`, inclusive | No | Existing AdvSTAD alternating generator/adversary objectives |
| `AdvSTAD` | `x[i]`, inclusive | Yes | Existing AdvSTAD alternating generator/adversary objectives |

## Dataset and windows

The existing SMD loader selects **`machine-1-1`**. The same fixed selection is
now shared through `src/study_protocol.py:SMD_MACHINE_ID`; there is no new
machine CLI argument or loop over machines. All four models read only:

```text
processed/SMD/machine-1-1_train.npy
processed/SMD/machine-1-1_test.npy
processed/SMD/machine-1-1_labels.npy
```

The existing train/test splits, observations, feature order, and preprocessing
are retained. `--less` still takes the central approximately 20% of training
observations only. Preprocessing itself remains unchanged; no preprocessing is
automatically launched by either the runner or comparison script.

Window length is 10 for both TranAD models. Both AdvSTAD variants use the same
`advstad.window_size` setting (currently 10). Windows have stride 1, repeat the
first observation at the start, never include future data, and use overlapping
views without modifying the input.

With length 3, the first three windows are:

```text
TranAD:            [x0,x0,x0], [x0,x0,x0], [x0,x0,x1]
All three others:  [x0,x0,x0], [x0,x0,x1], [x0,x1,x2]
```

Each model reconstructs its window endpoint, supplied as the decoder query.
Scores use decoder 2's per-feature squared error against that endpoint. The
three inclusive variants therefore align score `i` with observation and label
`i`. Original TranAD deliberately retains its legacy endpoint/label mismatch
for comparison A. Its plot-only observation roll is also retained; the corrected
variant does not roll observations. Train-score calibration uses each variant's
same window policy as test scoring, including `--test` runs.

## Architecture and interpretation

`TranAD_Corrected` subclasses TranAD without adding parameters or changing its
forward pass, loss, optimizer, scheduler, batch size, or training duration. Only
its runner windowing and plot alignment change. Both use the original single
AdamW optimizer and StepLR scheduler. Comments in the baseline do not constitute
additional adversarial optimization or MAML; the actual existing loss is kept.

`AdvSTAD_NoSpatial` starts from AdvSTAD's components and retains the temporal
encoder, positional encoding, query projection, both reconstruction decoders,
independent output heads, conditioning residual, two optimizers, two schedulers,
and alternating objectives. No spatial input projection, sensor embedding,
spatial encoder, spatial dropout, or fusion parameters are constructed. Temporal
memory is passed directly to the decoders at the existing `d_model` width. No
replacement projection or dummy spatial output is used.

In full AdvSTAD, sum and concat align spatial tokens to time tokens; concat also
projects the concatenation back to `d_model`, and cross-attention queries the
spatial tokens from temporal tokens. All these spatial-dependent operations are
bypassed by the ablation. Shared YAML settings remain accepted, but the no-spatial
model records its effective configuration as `fusion: temporal_only` and
`spatial_layers: 0`; changing unused fusion/spatial-depth settings does not create
a different no-spatial checkpoint. Continue using the existing supported fusion
values in `config.yaml`, which is also used by full AdvSTAD.

The no-spatial model is structurally close to corrected TranAD at the default
dimensions. It is **not identical**: AdvSTAD retains separate decoder output
heads and alternating adversarial updates, whereas TranAD shares its output
head and uses the existing joint reconstruction objective.

For 38 SMD features and the current configuration (window 10, `d_model=76`,
38 heads, one layer per stack, feedforward 16, concat fusion):

| Model | Trainable parameters |
| --- | ---: |
| TranAD | 127,538 |
| TranAD_Corrected | 127,538 |
| AdvSTAD_NoSpatial | 130,464 |
| AdvSTAD | 176,242 |

These counts come from initialization only, not dataset inference. Full AdvSTAD
counts change with fusion or dimensions. All runs record their actual counts.

Interpret the three requested comparisons as follows:

- **A: TranAD vs TranAD_Corrected:** isolate timestamp correction, retaining the
  legacy mismatch intentionally in the original baseline.
- **B: AdvSTAD_NoSpatial vs AdvSTAD:** isolate removal of the independent spatial
  route and its dependent fusion. Match all active temporal and training settings.
- **C: TranAD_Corrected vs both AdvSTAD variants:** compare overall architectures;
  this includes differences in heads, objectives, clipping, and update counts.

## Manual execution

Run from the project root in the environment containing the installed
dependencies. These commands are provided for manual execution and were not
launched as part of implementing the study.

Fresh training, followed by the runner's existing evaluation phase:

```bash
python main.py --model TranAD --dataset SMD --retrain --run-name ablation-tranad
python main.py --model TranAD_Corrected --dataset SMD --retrain --run-name ablation-corrected
python main.py --model AdvSTAD_NoSpatial --dataset SMD --retrain --run-name ablation-no-spatial
python main.py --model AdvSTAD --dataset SMD --retrain --run-name ablation-full
```

Evaluate each matching saved checkpoint independently:

```bash
python main.py --model TranAD --dataset SMD --test --run-name ablation-tranad-test
python main.py --model TranAD_Corrected --dataset SMD --test --run-name ablation-corrected-test
python main.py --model AdvSTAD_NoSpatial --dataset SMD --test --run-name ablation-no-spatial-test
python main.py --model AdvSTAD --dataset SMD --test --run-name ablation-full-test
```

Omitting both `--retrain` and `--test` resumes a matching checkpoint for another
invocation's epochs. Original defaults remain five epochs for TranAD and its
corrected variant, and `advstad.training.epochs_per_run` for both AdvSTAD
variants. Check total completed epochs when comparing resumed runs.

`--test` requires a matching, readable checkpoint containing model weights and
at least one completed epoch. Missing, corrupt, untrained, or incompatible
checkpoints fail clearly, including when `--retrain` is also supplied.
`--run-name` labels tracking records and does not choose checkpoint weights.
Before the first checkpoint write in a later training invocation, the previous
checkpoint is copied to a unique `archive/model-<uuid>.ckpt` in its model folder.
Subsequent epochs update the usual `model.ckpt` resume path. The archive path is
recorded in the new run's artifacts. No existing checkpoint is rewritten during
configuration checks or tests, which use temporary fixtures only.

## Checkpoints and experiment records

Checkpoint locations:

```text
checkpoints/TranAD_SMD/model.ckpt
checkpoints/TranAD_Corrected_SMD/model.ckpt
checkpoints/AdvSTAD_NoSpatial_SMD_temporal_only_<fingerprint>/model.ckpt
checkpoints/AdvSTAD_SMD_<fusion>_<fingerprint>/model.ckpt
```

Original AdvSTAD fingerprints and checkpoint schemas remain compatible. The
no-spatial metadata includes its model identity and effective architecture, so
full/ablated weights cannot be silently interchanged. Corrected TranAD adds
model/alignment metadata and refuses legacy TranAD payloads in its checkpoint
path. Existing original checkpoints are not converted or rewritten.

With experiment tracking enabled, each invocation creates a separate
`runs/<run-id>/run.json`, per-feature CSV, and TensorBoard directory. The actual
root is `experiment_tracking.log_dir` in `config.yaml`. Completed and failed
records still append to `experiments.jsonl`; old records are not migrated.
Run IDs include a UUID suffix to prevent timestamp collisions. New plots use
`plots/<experiment-key>/<run-id>/`, preserving the plot files referenced by
previous records instead of replacing shared plot outputs.

Existing record fields are preserved. Additions include:

- `seed` and `reproducibility`: actual Python, NumPy, PyTorch/CUDA seeds and
  deterministic backend flags. Root `seed: 1` preserves the original PyTorch
  seed and call position; Python and NumPy now receive the same seed. PyTorch's
  `manual_seed` also seeds CUDA generators lazily. Checkpoint RNG restoration
  is not introduced, and runtime code does not pretend to change the process's
  `PYTHONHASHSEED` after startup.
- `dataset_details.machine_id`, input-file SHA256 `data_fingerprints`, source
  paths, shapes, and window alignment. Hashes identify the unchanged processed
  train/test/label files; the separately recorded `less_data` flag identifies
  any central training subset.
- `evaluation_protocol`: the existing score, mean sensor aggregation, label
  reduction, point adjustment, binary ROC/AUC convention, and POT parameters.
- Active resolved architecture, objective identifiers, parameter counts,
  optimizer/scheduler settings, source revision, device, CPU/GPU, precision,
  PyTorch version, thread count, timestamps, and checkpoint identity.
- `hyperparameters.effective_config` records the constructed architecture for
  every model, including TranAD's width, heads, depths, dropout, output heads,
  training objective, clipping, and applicable epoch count. Optimizer records
  include actual betas, epsilon, and AMSGrad settings. Raw shared YAML remains
  separately available under `configuration`; its `advstad` block does not
  configure either TranAD variant.
- `hyperparameters.training_seed` identifies the checkpoint's training seed
  when known. Newly saved payloads include this optional provenance field;
  loading one with a different selected seed fails. Older payloads remain
  compatible but have an unknown training seed rather than an invented value.
- `metrics.runtime`: the timing values below and current/cumulative epoch counts.
- `metrics.memory`: available CUDA allocation/reservation and process peak RSS
  in bytes, recorded consistently for all models, including failures where possible.

## Runtime boundaries

All new timings use monotonic `time.perf_counter()` with CUDA synchronization
immediately before and after each measured GPU interval. Measurements use the
same code across all four models and retain the selected optimization settings.

| `metrics.runtime` field | Units and boundary |
| --- | --- |
| `training_total_seconds` | Sum of completed training-epoch calls in this invocation; excludes loading, windowing, checkpoint writes, and tracking |
| `training_average_epoch_seconds` | Training compute total divided by completed epochs in this invocation |
| `inference_total_seconds` | Test scoring only, including batching, forward passes, reconstruction errors, and CPU output transfer |
| `inference_latency_ms_per_timepoint` | Test scoring seconds × 1000 / scored test timestamps; amortized batched latency, not single-request latency |
| `training_epochs_this_run` | Completed training epochs measured in this invocation |
| `training_epochs_total` | Completed checkpoint epochs plus this invocation's completed epochs |

Inference excludes data loading, provenance hashing, window creation, training
score calibration, plotting, POT, and metric computation. It runs with `eval()`
and no gradients. The timing version is `synchronized_perf_counter_v1`.

The existing broad `metrics.timing_seconds.training` and `evaluation` fields
remain available, now using the monotonic clock. They retain their broader
phase boundaries. The comparison script never substitutes broad evaluation time
for inference time. Test-only runs have unavailable training durations and zero
epochs trained in that invocation; historical training costs are not invented
or borrowed from other records.

Memory values are **process-lifetime peaks**, not isolated model-forward peaks:
they can include data loading, CUDA initialization, training, plotting, and metric
work. Report this scope when using them. CSV memory uses decimal MB (1,000,000
bytes), while JSON preserves bytes.

## Compare saved results

```bash
python ablation_study.py --compare
```

This command imports no models or training runner and performs no preprocessing,
checkpoint inference, training, or evaluation. It reads only
`<configured-log-dir>/*/run.json`, prints a table and warnings, and generates:

```text
ablation_results/SMD/machine-1-1/ablation_comparison.csv
```

The CSV has exactly these five columns:

```text
Metric,TranAD,TranAD_Corrected,AdvSTAD_NoSpatial,AdvSTAD
```

Rows include precision, recall, F1, existing additional metrics, measured compute
costs, epoch counts, trainable parameters, available memory, machine/run IDs,
seed, windowing, checkpoint, timing version, and device/precision. Accuracy is
included only if it already exists in the selected record. No accuracy metric
has been added to the evaluation pipeline.

Only completed SMD records explicitly identifying `machine-1-1` are eligible.
Historical records without a machine ID are excluded with an uncertainty
warning; their array shapes alone cannot prove machine identity. Failed,
incomplete, other-machine, malformed, or missing records never supply results.

The deterministic selection policy maximizes the number of verified compatible
model columns. Ties use newest completion time, then run ID and path, with the
newest matching run selected per model. Controls include data fingerprints,
shapes, seed/backend flags, device/hardware/precision, PyTorch/thread count,
window/batch/base learning rate, optimizer/scheduler settings, total trained
epochs, training subset, evaluation protocol, and timing version. The spatial
pair additionally requires matching active temporal and adversarial settings.
Expected window-alignment and model-objective differences remain study factors.

Pin exact records or filter a seed when needed:

```bash
python ablation_study.py --compare --seed 1 \
  --run TranAD=<run-id> \
  --run TranAD_Corrected=<run-id> \
  --run AdvSTAD_NoSpatial=<run-id> \
  --run AdvSTAD=<run-id>
```

Replace the angle-bracket placeholders with actual run IDs. An incompatible
explicit selection is not combined: the first valid pin anchors the comparison,
and conflicting columns become `N/A` with warnings. If other required metadata
is unknown, at most one clearly marked unverified reference is shown. Missing
model results remain `N/A`; no values are fabricated or merged across runs.
An all-missing report is valid before the new models have been executed.

`--runs-dir PATH` overrides discovery, and `--output-dir PATH` writes directly
into that report directory. Relative overrides use the working directory. The
script may update its comparison CSV but never modifies input `run.json` files.

## Fairness limits and lightweight validation

The existing metrics remain point-adjusted using test labels. `ROC/AUC` remains
the legacy value from adjusted binary predictions, not continuous-score AUROC.
No thresholding definitions or test-label adjustments were changed.

Same seed does not imply identical shared weights across different architectures:
parameter construction consumes RNG differently. All library RNGs are now seeded,
but deterministic algorithms are not newly enabled and resumed training does
not restore RNG states. CUDA/cuDNN determinism and TF32 flags remain at their
existing settings and are recorded; changing them would be a separate scientific
decision. Repeated seeds and stronger reproducibility controls would likewise be
an experimental decision. Equal epochs do not equal equal
computation, especially with AdvSTAD's alternating optimizer steps. Source
revision differences and uncommitted changes are reported as provenance limits.

The current project requests CUDA and float32, with no AMP, explicit GPU memory
cap, or project allocator override. Device selection falls back to CPU only when
CUDA is unavailable. Tracking records the selected device, physical GPU memory,
external allocator/visible-device environment, and precision backend flags.
Do not compare CPU and CUDA timings as if they used the same hardware.

No epoch settings or scheduling behavior were changed by the reproducibility
checks: TranAD remains five epochs per invocation; both AdvSTAD variants keep
their configured `epochs_per_run`, absolute-epoch objective schedules, and
existing StepLR behavior. SMD machine selection, saved arrays, window alignment,
losses, concat fusion, module construction order, and checkpoint fingerprints
are preserved.

Use matching hardware and configuration, avoid overlapping training processes,
and retain total epoch counts. Training timings cover the selected invocation,
so a resumed or test-only run does not provide the full cost of its checkpoint.
The script flags these limitations without silently combining different seeds
or incompatible settings.

Validation uses small synthetic arrays and temporary records/checkpoints only:

```bash
python -m pytest -q tests
```

Tests cover boundary padding, no future observations, all four reconstruction
endpoints, actual tiny forward/backward updates, disjoint optimizer ownership,
checkpoint isolation, original-model preservation, tracking/failure records,
timing boundaries, and saved-results comparison. No real SMD training or scoring
is launched by the test fixtures.
