#!/usr/bin/env python3
"""Sample RAM, CPU, CUDA and Python stacks without changing application code.

Watch an existing process (no Python stack access is needed):
    python tools/profile_resources.py watch --pid 1234 --seconds 30

Record a future run, including main-thread stacks and tensor/array sizes:
    venv/bin/python tools/profile_resources.py run --output runs/resources.jsonl \
        -- main.py --model TranAD --dataset SWaT --retrain

The run command executes the application's normal entry point and therefore
produces its normal checkpoints, plots and experiment records. Sampling adds
overhead; CUDA memory counters do not synchronize GPU work. GPU utilization from
nvidia-smi is device-wide, not attributable to a particular process under WSL.
Only the Python standard library is required by this tool itself (Linux /proc).
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading
import time


def proc_memory(pid):
    wanted = {"VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem", "VmSwap"}
    result = {}
    with open("/proc/{}/status".format(pid)) as handle:
        for line in handle:
            key, _, value = line.partition(":")
            if key in wanted:
                result[key + "_bytes"] = int(value.split()[0]) * 1024
    return result


def proc_cpu_seconds(pid):
    # The process name can contain spaces or parentheses.
    fields = Path("/proc/{}/stat".format(pid)).read_text().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def gpu_sample():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=True,
        )
        return result.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError) as error:
        return {"unavailable": str(error)}


def describe(value):
    """Only describe arrays/tensors; never copy their contents or compute reductions."""
    module = type(value).__module__
    if module.startswith("numpy") and hasattr(value, "shape"):
        return {"type": "numpy", "shape": list(value.shape), "dtype": str(value.dtype),
                "logical_bytes": int(value.nbytes)}
    torch = sys.modules.get("torch")
    if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(value):
        return {"type": "tensor", "shape": list(value.shape), "dtype": str(value.dtype),
                "device": str(value.device),
                "logical_bytes": value.numel() * value.element_size(),
                "storage_bytes": value.storage().size() * value.element_size()}
    return None


def python_sample(thread_id, root):
    frame = sys._current_frames().get(thread_id)
    stack = []
    while frame is not None:
        path = Path(frame.f_code.co_filename).resolve()
        item = {"file": str(path), "line": frame.f_lineno,
                "function": frame.f_code.co_name}
        # Exclude third-party packages even when the virtualenv lives in root.
        if path == root / "main.py" or root / "src" in path.parents:
            values = {}
            for name, value in list(frame.f_locals.items()):
                info = describe(value)
                if info is not None:
                    values[name] = info
                elif name in {"epoch", "e", "i", "dim", "offset", "lms", "training"}:
                    if isinstance(value, (int, float, bool, str, type(None))):
                        values[name] = value
            item["locals"] = values
        stack.append(item)
        frame = frame.f_back
    result = {"main_thread_stack_innermost_first": stack}
    torch = sys.modules.get("torch")
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_initialized():
        result["cuda_bytes"] = {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
        }
    return result


class Sampler:
    def __init__(self, pid, output, interval, thread_id=None, root=None):
        self.pid, self.output, self.interval = pid, output, interval
        self.thread_id, self.root = thread_id, root
        self.stop = threading.Event()

    def write(self, record):
        self.output.write(json.dumps(record, allow_nan=False) + "\n")
        self.output.flush()

    def sample(self):
        last_cpu, last_time = None, None
        while not self.stop.is_set():
            now = time.monotonic()
            try:
                cpu = proc_cpu_seconds(self.pid)
                record = {"event": "sample", "pid": self.pid,
                          "time": datetime.datetime.now().astimezone().isoformat(),
                          "memory": proc_memory(self.pid)}
            except (FileNotFoundError, ProcessLookupError):
                return
            if last_time is not None:
                record["cpu_percent_one_core_100"] = 100 * (cpu - last_cpu) / (now - last_time)
            last_cpu, last_time = cpu, now
            if self.thread_id is not None:
                try:
                    record.update(python_sample(self.thread_id, self.root))
                except Exception as error:
                    record["stack_sample_error"] = repr(error)
            record["gpu_device_wide_csv"] = gpu_sample()
            self.write(record)
            self.stop.wait(self.interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("watch", "run"))
    parser.add_argument("--pid", type=int)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--output", help="JSONL path; defaults to stdout in watch mode")
    # Parse tool options independently of the target command after --.
    own_args, separator, command = list(sys.argv[1:]), False, []
    if "--" in own_args:
        split = own_args.index("--")
        command, own_args, separator = own_args[split + 1:], own_args[:split], True
    options = parser.parse_args(own_args)
    if options.interval <= 0 or options.seconds <= 0:
        parser.error("--interval and --seconds must be positive")
    if options.mode == "watch" and (options.pid is None or command):
        parser.error("watch requires --pid and does not accept a command")
    if options.mode == "run" and (not separator or not command or not options.output):
        parser.error("run requires --output and -- script.py [arguments]")
    output = sys.stdout
    if options.output:
        path = Path(options.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        output = path.open("x", encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    pid = options.pid if options.mode == "watch" else os.getpid()
    sampler = Sampler(pid, output, options.interval,
                      threading.get_ident() if options.mode == "run" else None, root)
    worker = threading.Thread(target=sampler.sample, name="resource-sampler", daemon=True)
    try:
        sampler.write({"event": "start", "mode": options.mode, "pid": pid,
                       "command": command, "interval_seconds": options.interval})
        worker.start()
        if options.mode == "watch":
            worker.join(options.seconds)
        else:
            script = str(Path(command[0]).resolve())
            sys.argv = [script] + command[1:]
            sys.path.insert(0, str(Path(script).parent))
            runpy.run_path(script, run_name="__main__")
    finally:
        sampler.stop.set()
        worker.join(timeout=3)
        sampler.write({"event": "stop", "pid": pid})
        if output is not sys.stdout:
            output.close()


if __name__ == "__main__":
    main()
