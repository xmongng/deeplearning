import argparse
import time

import torch

from .common import ROOT, config, seed_all, read_csv, write_csv, write_json, environment
from .data import loader
from .export_edge import export_model, load_fp32
from .models import NAMES
from .quantization import convert, prepare


def main():
    p = argparse.ArgumentParser(description="Static W8A8 PTQ calibrated only on the fixed train subset")
    p.add_argument("--config")
    p.add_argument("--model", required=True, choices=NAMES)
    args = p.parse_args()
    cfg = config(args.config)
    seed_all(cfg["seed"])
    torch.set_num_threads(4)
    model, meta = load_fp32(cfg, args.model)
    started = time.perf_counter()
    prepared = prepare(model, cfg)
    calibration_log = []
    evidence = ROOT / "results/quantization" / args.model / "ptq"
    write_json(evidence / "run_config.json", {"config": cfg, "environment": environment(), **meta,
               "scheme": "static W8A8; symmetric per-channel weights; XNNPACK PT2E",
               "calibration_split": "calibration (train-only)", "device": "cpu"})
    with torch.no_grad():
        for index, (x, y, paths) in enumerate(loader(cfg, "calibration")):
            batch_start = time.perf_counter()
            prepared(x)
            calibration_log.append({"index": index + 1, "path": paths[0], "label": y.item(),
                                    "seconds": time.perf_counter() - batch_start})
            write_csv(evidence / "calibration.csv", calibration_log)
            if (index + 1) % 25 == 0:
                print(f"{args.model} PTQ calibrated {index + 1} images", flush=True)
    (evidence / "prepared_graph.txt").write_text(str(prepared) + "\n")
    observer_state = {key: value.detach().cpu() for key, value in prepared.state_dict().items()
                      if "activation_post_process" in key or "observer" in key}
    torch.save(observer_state, evidence / "observer_state.pth")
    model = convert(prepared)
    meta.update(calibration_paths=[r["path"] for r in read_csv(ROOT / "data/splits/calibration.csv")],
                ptq_seconds=time.perf_counter() - started)
    report = export_model(model, cfg, args.model, "ptq", meta)
    write_json(evidence / "summary.json", {**report, "calibration_images": len(calibration_log)})


if __name__ == "__main__":
    from .stage_runner import logged_main
    logged_main("ptq_int8", main)
