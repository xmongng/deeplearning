"""Fresh-process ExecuTorch/XNNPACK CPU benchmark with verified native threads."""
from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import threading
import time

import numpy as np
import psutil
import torch

from .common import ROOT, config, environment, seed_all, write_csv, write_json
from .evaluate import verify_artifact
from .export_edge import load_runtime
from .quantization import example


def set_threads(count):
    # torch.set_num_threads alone does NOT control ExecuTorch's own pool.
    from executorch.extension.pybindings.portable_lib import (
        _threadpool_get_thread_count, _unsafe_reset_threadpool)
    torch.set_num_threads(count)
    torch.set_num_interop_threads(1)
    _unsafe_reset_threadpool(count)
    actual = _threadpool_get_thread_count()
    if actual != count:
        raise RuntimeError(f"Requested {count} ExecuTorch threads, got {actual}")
    return actual


def worker(cfg, name, method, threads, repeat, output):
    actual = set_threads(threads)  # before loading any model; safe in a fresh process
    path, meta = verify_artifact(cfg, name, method)
    seed_all(cfg["seed"])
    x = example(cfg)
    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    start = time.perf_counter_ns()
    program, forward = load_runtime(path)
    load_ms = (time.perf_counter_ns() - start) / 1e6
    rss_loaded = proc.memory_info().rss
    b = cfg["benchmark"]
    latencies = []
    with torch.inference_mode():
        for _ in range(b["warmup"]):
            forward.execute((x,))
        for _ in range(b["iterations"]):
            start = time.perf_counter_ns()
            forward.execute((x,))
            latencies.append((time.perf_counter_ns() - start) / 1e6)
        # Sample memory in a separate untimed pass; sampling does not perturb latency.
        peak = [proc.memory_info().rss]
        stop = threading.Event()

        def sample():
            while not stop.is_set():
                peak[0] = max(peak[0], proc.memory_info().rss)
                stop.wait(0.001)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        try:
            for _ in range(b["iterations"]):
                forward.execute((x,))
        finally:
            stop.set()
            sampler.join()
        peak[0] = max(peak[0], proc.memory_info().rss)
    raw_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = raw_peak if platform.system() == "Darwin" else raw_peak * 1024
    result = {"model": name, "method": method, "threads": actual, "repeat": repeat,
              "median_ms": float(np.median(latencies)), "p95_ms": float(np.percentile(latencies, 95)),
              "load_ms": load_ms, "rss_before_load_mb": rss_before / 1e6,
              "rss_loaded_mb": rss_loaded / 1e6, "sampled_peak_inference_rss_mb": peak[0] / 1e6,
              "peak_process_rss_mb": peak_bytes / 1e6,
              "latencies_ms": latencies, "serialized_size_mb": meta["serialized_size_mb"],
              "pte_sha256": meta["pte_sha256"], "backend": "XNNPACK", "runtime": "ExecuTorch",
              "environment": environment(), "benchmark_config": b,
              "memory_definition": "Decimal MB; RSS of Python process including imports/runtime. Inference RSS sampled every 1 ms in separate untimed pass; may miss short peaks. ru_maxrss is lifetime process high-water mark, including imports/load."}
    write_json(output, result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--model")
    p.add_argument("--method", choices=["fp32", "ptq", "qat"])
    p.add_argument("--threads", type=int)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--output")
    p.add_argument("--conditions", default="Not recorded", help="Power, background apps, thermal state, room temperature")
    args = p.parse_args()
    cfg = config(args.config)
    if args.worker:
        worker(cfg, args.model, args.method, args.threads, args.repeat, args.output)
        return
    for name in cfg["models"]:
        for method in ("fp32", "ptq", "qat"):
            verify_artifact(cfg, name, method)
    seed_all(cfg["seed"])
    import random
    jobs = [(n, m, t, r) for n in cfg["models"] for m in ("fp32", "ptq", "qat")
            for t in cfg["benchmark"]["threads"] for r in range(cfg["benchmark"]["repeats"])]
    random.shuffle(jobs)
    raw = []
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = ROOT / "results/raw_logs" / stamp
    log_dir.mkdir(parents=True)
    write_json(log_dir / "conditions.json", {"conditions": args.conditions, "environment": environment(),
                                             "job_order": jobs, "config": cfg})
    for name, method, threads, repeat in jobs:
        output = log_dir / f"{name}_{method}_t{threads}_r{repeat}.json"
        cmd = [sys.executable, "-m", "src.benchmark", "--worker", "--model", name, "--method", method,
               "--threads", str(threads), "--repeat", str(repeat), "--output", str(output)]
        if args.config:
            cmd += ["--config", str(__import__("pathlib").Path(args.config).resolve())]
        with output.with_suffix(".log").open("w") as f:
            subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
        row = json.loads(output.read_text())
        raw.append(row)
        print(f"{name}/{method}, threads={threads}, repeat={repeat}: {row['median_ms']:.3f} ms", flush=True)
    rows = []
    for name in cfg["models"]:
        for threads in cfg["benchmark"]["threads"]:
            baseline = next(r for r in raw if r["model"] == name and r["threads"] == threads and r["method"] == "fp32")
            base_runs = [r for r in raw if r["model"] == name and r["threads"] == threads and r["method"] == "fp32"]
            base_median = float(np.median([x for r in base_runs for x in r["latencies_ms"]]))
            for method in ("fp32", "ptq", "qat"):
                runs = [r for r in raw if r["model"] == name and r["threads"] == threads and r["method"] == method]
                values = [x for r in runs for x in r["latencies_ms"]]
                median = float(np.median(values))
                rows.append({"model": name, "method": method, "threads": threads,
                    "median_ms": median, "p95_ms": float(np.percentile(values, 95)),
                    "repeat_median_std_ms": float(np.std([r["median_ms"] for r in runs], ddof=1)) if len(runs)>1 else 0.0,
                    "peak_process_rss_mb": max(r["peak_process_rss_mb"] for r in runs),
                    "sampled_peak_inference_rss_mb": max(r["sampled_peak_inference_rss_mb"] for r in runs),
                    "load_median_ms": float(np.median([r["load_ms"] for r in runs])),
                    "serialized_size_mb": runs[0]["serialized_size_mb"], "speedup": base_median / median,
                    "compression_ratio": baseline["serialized_size_mb"] / runs[0]["serialized_size_mb"],
                    "pte_sha256": runs[0]["pte_sha256"], "device": platform.node(), "backend": "XNNPACK",
                    "raw_log_directory": str(log_dir.relative_to(ROOT))})
    write_csv(ROOT / "results/mac_m3_benchmark.csv", rows)


if __name__ == "__main__":
    from .stage_runner import logged_main
    # Worker stdout/stderr already has a dedicated raw log and JSON in the parent.
    if "--worker" in sys.argv:
        main()
    else:
        logged_main("benchmark", main)
