from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import functools
import platform
import random
import subprocess
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def config(path=None):
    return yaml.safe_load(Path(path or ROOT / "configs/project.yaml").read_text())


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", newline="", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_fingerprint():
    h = hashlib.sha256()
    for name in ("train", "val", "test", "calibration"):
        h.update((ROOT / f"data/splits/{name}.csv").read_bytes())
    return h.hexdigest()


@functools.lru_cache(maxsize=1)
def environment():
    versions = {}
    for name in ("torch", "torchvision", "torchao", "executorch", "numpy", "Pillow", "scikit-learn", "psutil"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {"platform": platform.platform(), "machine": platform.machine(),
              "python": platform.python_version(), "versions": versions,
              "mps_available": torch.backends.mps.is_available(),
              "cuda_available": torch.cuda.is_available(), "torch_cuda_build": torch.version.cuda,
              "cudnn_version": torch.backends.cudnn.version(), "torch_threads": torch.get_num_threads(),
              "gpus": [{"index": i, "name": torch.cuda.get_device_name(i),
                        "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                        "capability": list(torch.cuda.get_device_capability(i))}
                       for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []}
    if platform.system() == "Darwin":
        for cmd in (["sw_vers"], ["system_profiler", "SPHardwareDataType"]):
            output = subprocess.check_output(cmd, text=True, timeout=30)
            # Environment records must remain shareable with an experiment.
            private_fields = ("Serial Number", "Hardware UUID", "Provisioning UDID")
            result[cmd[0]] = "\n".join(
                line for line in output.splitlines()
                if not any(field in line for field in private_fields)
            ) + "\n"
    return result


def device_for(name):
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def artifact_dir(model, smoke=False):
    return ROOT / ("results/smoke/exported" if smoke else "exported") / model


def checkpoint_path(model):
    return ROOT / "checkpoints" / model / "fp32_best.pth"


def check_checkpoint(ckpt, cfg):
    if ckpt["classes"] != cfg["classes"]:
        raise ValueError("Checkpoint class mapping differs from config")
    if ckpt["split_fingerprint"] != split_fingerprint():
        raise ValueError("Dataset splits changed since training; start a new experiment")
    if ckpt["preprocessing"] != cfg["preprocessing"]:
        raise ValueError("Preprocessing changed since training")


def rng_state():
    """Save RNG state in weights_only-compatible types for later reproducibility audits."""
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": {"generator": numpy_state[0],
            "keys": numpy_state[1].tolist(), "position": numpy_state[2],
            "has_gauss": numpy_state[3], "cached_gaussian": numpy_state[4]},
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_state(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n['generator'], np.asarray(n['keys'], dtype=np.uint32),
                         n['position'], n['has_gauss'], n['cached_gaussian']))
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda']:
        if not torch.cuda.is_available() or len(state['torch_cuda']) != torch.cuda.device_count():
            raise ValueError('Resume requires the original CUDA device count for RNG restoration')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['torch_cuda']])
