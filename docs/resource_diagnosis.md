# SWaT RAM and GPU diagnosis — 2026-10-24

CUDA is being used. The investigation found a large host-memory cost during CUDA initialization, a reproducible infinite CPU retry in scoring, and substantial CPU work outside the models. Application code and configuration were not changed. The existing training process was left running.

The machine has approximately 7.7 GiB RAM, 2 GiB swap and a GTX 1660 Ti with 6 GiB VRAM. Both October 24 experiment records report `device=cuda`, `dtype=float32`, and PyTorch `1.8.1+cu111`. The configured model and observation tensors are explicitly moved to CUDA in `main.py:69`, `main.py:134`, and `main.py:139`. Labels deliberately remain in NumPy on the CPU.

## 1. CUDA initialization itself consumes several GiB of host RAM

A separate, bounded probe loaded the saved TranAD checkpoint and evaluated 4,096 real windows. Before CUDA initialization, the probe had only memory-mapped the training file; it had not materialized the dataset or performed inference. Its measured resident memory was:

| Probe stage | Host RSS | Live CUDA tensors |
| --- | ---: | ---: |
| NumPy and PyTorch imported | 0.96 GiB | CUDA not initialized |
| Repository imported; small model on CPU | 1.04 GiB | CUDA not initialized |
| Same model moved to configured CUDA device | 3.36 GiB | 0.87 MiB |
| Checkpoint loaded | 3.34 GiB | 0.87 MiB |
| 4,096-window inference completed | 3.46 GiB | 8.84 MiB |

The probe's host RSS peaked at 3.61 GiB, while peak live CUDA tensors during inference were only 16.34 MiB. The transition to CUDA increased anonymous host RAM from 0.32 GiB to 2.87 GiB. This isolates a major RAM cost to CUDA initialization/runtime activity rather than SWaT windows, optimizer state, or model weights. Importing PyTorch also mapped substantial file-backed library memory, which is more reclaimable than anonymous memory.

The exact allocation inside the CUDA runtime/driver was not identified. Eager loading of the old CUDA libraries is a plausible contributor: NVIDIA documents that lazy loading reduces both host and GPU memory overhead, requires CUDA runtime 11.7+, and does not benefit older runtime libraries just because the driver is new. This environment uses CUDA 11.1 through its PyTorch build. That explains why a modern driver alone does not provide those runtime benefits. See [NVIDIA's lazy-loading requirements](https://docs.nvidia.com/cuda/archive/12.1.1/cuda-c-programming-guide/lazy-loading.html).

Evidence: `runs/resource-diagnosis/checkpoint-probe.json` and `checkpoint-profile.jsonl`. The probe reports allocator memory, whereas `nvidia-smi` includes additional CUDA context/library overhead. These are different measurements.

## 2. POT scoring can retry forever with constant sensor scores

`main.py:478` calls `pot_eval` separately for all 51 sensors. In `src/pot.py:134–141`, a `while True` catches every exception from SPOT initialization, multiplies the threshold level by `0.999`, and retries without a limit or diagnostic message.

SPOT selects peaks strictly greater than its initialization threshold (`src/spot.py:195–200`). If the training scores are constant, there are no peaks at any threshold level selected from those scores. `_grimshaw` then calls `self.peaks.min()` at `src/spot.py:332`, raising:

```text
ValueError: zero-size array to reduction operation minimum which has no identity
```

The retry cannot fix this input. Each iteration repeats CPU sorting and calibration setup, so one CPU core remains busy and the GPU has no work.

The actual training array has 11 constant columns. Using the saved epoch-4 TranAD checkpoint, all 11 also produced constant reconstruction scores across 4,096 real windows sampled throughout the training timeline:

```text
P102 P201 P202 P204 P206 P401 P403 P404 P502 P601 P603
```

The sampled scores were exactly zero, except P204, whose score was exactly one. SPOT initialization failed with the above exception for each. P102 is feature index 4, so a problematic feature occurs early in per-sensor evaluation. These are checkpoint-sample observations, not an exhaustive inference pass over every training window.

A separate small reproduction called the original `pot_eval` with constant scores. It did not return; the sampler repeatedly captured `src/pot.py:139` at approximately 95–99% of one CPU core, with `lms` shrinking from roughly `0.0109` to `1.2e-13`. Only this diagnostic child was terminated after four seconds. Evidence: `runs/resource-diagnosis/constant-pot-profile.jsonl`.

The existing TranAD process, PID 1572 at investigation time, had completed all five training epochs in 505.43 seconds and recorded its finished evaluation PDF before the investigation. It was then observed using approximately one CPU core with 0% GPU utilization. This is consistent with the reproduced scoring failure. Linux denied access to its process memory, so its exact live Python stack was not directly captured; the attribution of that particular process to the retry loop is an inference from its completed stages and the reproduction.

The AdvSTAD record shows a KeyboardInterrupt after one completed epoch. It did not reach the scoring stage in that recorded run, so the POT loop does not explain that run's training-time behavior. Both models share the scoring path and CUDA stack.

## 3. The new dataset and CPU evaluation work are much larger

The previous loader, visible in commit `d4bf8ae^`, selected only 3,000 training rows and 5,000 test rows of the JSON `val` column. Current preprocessing retains 495,000 training timestamps and 449,919 test timestamps across 51 sensors, with `downsample=1`. That is 165 times as many training timestamps and approximately 90 times as many test timestamps, plus the wider sensor input.

Current on-disk arrays are:

| Array | Shape | Disk dtype | Size |
| --- | --- | --- | ---: |
| train | 495,000 × 51 | float64 | 192.60 MiB |
| test | 449,919 × 51 | float64 | 175.06 MiB |
| labels | 449,919 × 51 | uint8 | 21.88 MiB |

Observation tensors are converted to the configured float32 on CUDA, so disk float64 does not mean the model is running in float64. Window conversion already uses an overlapping `as_strided` view (`main.py:27–48`); it does not materialize a tenfold copy of all windows.

Both model evaluation paths explicitly copy every batch's scores and predictions to CPU and retain them until concatenation (`main.py:375–385`, `src/advstad_training.py:181–195`). Train and test scores/predictions together amount to approximately 368 MiB at float32, before batch-list/concatenation overlap, labels, plot data, or library memory. Even the unused training predictions remain referenced by `_` in `run_experiment`.

`src/plotting.py:19–41` renders all 449,919 timestamps for each of 51 sensors using Matplotlib, including multiple curves and a filled label region. This is CPU work with host allocations. The actual TranAD output PDF was approximately 163 MiB. Its exact contribution to peak RAM was not isolated. The existing process's initial observed RSS was about 4.72 GiB and its recorded peak RSS was 5.54 GiB, so CUDA initialization alone does not account for its full peak.

POT, NumPy/scikit-learn metrics, plotting and experiment serialization all execute on CPU irrespective of `device: cuda`. Completed plots do not indicate that scoring has completed.

## 4. Training also contains frequent CPU/GPU synchronization

TranAD extracts a Python loss with `.item()` on every batch (`main.py:366`). AdvSTAD extracts seven metric scalars per alternating training step (`src/advstad_training.py:148–153`), and performs two optimizer updates. Both models have short sequences and small parameter counts: 225,519 for TranAD and 310,766 for this AdvSTAD configuration. A batch occupies the GPU briefly; CPU launch work and scalar reads can leave gaps between kernels.

CPU copies and scalar extraction can synchronize CUDA execution, as documented in [PyTorch's CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html). This is a code-level explanation for potential training overhead, not a measured breakdown of the AdvSTAD epoch. A sampled utilization of zero alone cannot establish that a model is on CPU.

## Diagnostic tools added

`tools/profile_resources.py` samples RSS, anonymous/file-backed RAM, swap, CPU usage and device-wide NVIDIA metrics. In `run` mode it also records the main Python thread's stack, selected loop variables, tensor devices, array sizes, underlying tensor storage sizes, and CUDA allocator counters. It does not import PyTorch itself, change application functions, install tracing hooks, synchronize CUDA, or write application configuration. A long native call holding Python's GIL can delay its sampling thread, so boundary readings in the checkpoint probe supplement the samples.

Watch an existing process, substituting the current PID and a fresh output path:

```bash
python3 tools/profile_resources.py watch --pid 1572 --seconds 30 \
  --output runs/resource-diagnosis/watch-next.jsonl
```

Profile a future normal run. This command really runs the application, including its normal checkpoint and plot writes:

```bash
venv/bin/python tools/profile_resources.py run \
  --output runs/resource-diagnosis/tranad-next.jsonl -- \
  main.py --model TranAD --dataset SWaT --retrain
```

`tools/probe_swat_resources.py` reads a trusted checkpoint, samples actual SWaT windows with the model's original endpoint alignment, runs evaluation, and tries SPOT initialization once per constant score column. It performs no training and writes only its diagnostic report. CUDA initialization still has the measured multi-GiB RAM cost even with few sampled windows; avoid running it alongside a memory-constrained training job.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg \
venv/bin/python tools/probe_swat_resources.py --model TranAD \
  --checkpoint checkpoints/TranAD_SWaT/model.ckpt \
  --output runs/resource-diagnosis/checkpoint-next.json
```

Validation performed: both CLIs and syntax checks passed; the watch mode captured the existing process; run mode captured memory and Python stacks during the actual CUDA checkpoint probe and the bounded constant-score retry reproduction. No full retraining was launched. The probe and reproduction outputs are under the already-ignored `runs/resource-diagnosis/` directory.
