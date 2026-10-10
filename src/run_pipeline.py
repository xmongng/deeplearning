"""Run the full experiment after CP0 and frozen data splits are available."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from .common import ROOT, config, environment, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config")
    p.add_argument("--conditions", default="Not recorded")
    args = p.parse_args()
    cfg = config(args.config)
    for name in cfg["models"]:
        status_file = ROOT / f"results/cp0/{name}.json"
        if not status_file.exists() or json.loads(status_file.read_text())["status"] != "passed":
            raise RuntimeError(f"Run CP0 smoke test successfully for {name} first")
    summary = ROOT / "data/splits/summary.json"
    if not summary.exists():
        raise RuntimeError("Audit and freeze data splits first")
    write_json(ROOT / "environment.json", environment())
    log_dir = ROOT / "results/pipeline_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def run(module, *extra):
        cmd = [sys.executable, "-u", "-m", "src.stage_runner", module, *extra]
        if args.config:
            cmd += ["--config", str(Path(args.config).resolve())]
        print("Running:", " ".join(cmd), flush=True)
        logfile = log_dir / (module + "_" + "_".join(extra).replace("/", "_") + ".log")
        with logfile.open("w") as f:
            process = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end="", flush=True)
                f.write(line)
                f.flush()
            if process.wait():
                raise RuntimeError(f"{module} failed; inspect {logfile}")

    for name in cfg["models"]:
        if not (ROOT / f"checkpoints/{name}/fp32_best.pth").exists():
            run("train_fp32", "--model", name)
        elif not (ROOT / f"checkpoints/{name}/training_summary.json").exists():
            raise RuntimeError(f"{name} training was interrupted. Archive its partial checkpoint directory before restarting; do not treat partial training as complete.")
        for method, module in (("fp32", "export_edge"), ("ptq", "ptq_int8"), ("qat", "qat_int8")):
            if not (ROOT / f"exported/{name}/{method}.json").exists():
                extra = ["--export-only"] if method == "qat" and (ROOT / f"checkpoints/{name}/qat_summary.json").exists() else []
                run(module, "--model", name, *extra)
    run("evaluate")
    run("benchmark", "--conditions", args.conditions)
    run("report", "--threads", str(cfg["benchmark"]["threads"][0]))


if __name__ == "__main__":
    from .stage_runner import logged_main
    logged_main("run_pipeline", main)
