"""Train all 9 configurations on Kaggle and package reproducible artifacts."""
import argparse
import shutil
import subprocess
import sys

import torch

from .common import ROOT, config, environment, write_json


def run(module, config_path, *args):
    cmd = [sys.executable, "-u", "-m", f"src.{module}", "--config", config_path, *args]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/kaggle.yaml")
    parser.add_argument("--skip-install-check", action="store_true")
    args = parser.parse_args()
    cfg = config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. In Kaggle select Settings > Accelerator > GPU.")
    print("GPU:", torch.cuda.get_device_name(0))
    write_json(ROOT / "environment-kaggle.json", environment())
    run("verify_data", args.config)
    for name in cfg["models"]:
        run("train_fp32", args.config, "--model", name, "--device", "cuda")
        run("export_edge", args.config, "--model", name)
        run("ptq_int8", args.config, "--model", name)
        run("qat_int8", args.config, "--model", name, "--device", "cuda")
    run("evaluate", args.config)
    # Package checkpoints, deploy artifacts, metrics and frozen protocol.
    package = ROOT / "kaggle-output"
    package.mkdir(exist_ok=True)
    for name in ("checkpoints", "exported", "results", "data", "configs", "environment-kaggle.json"):
        source = ROOT / name
        target = package / name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns("raw"))
        elif source.exists():
            shutil.copy2(source, target)
    shutil.make_archive("/kaggle/working/waste-edge-complete", "zip", package)
    print("Created /kaggle/working/waste-edge-complete.zip")


if __name__ == "__main__":
    main()
