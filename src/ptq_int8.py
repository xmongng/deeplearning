import argparse
import time

import torch

from .common import ROOT, config, seed_all, read_csv
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
    with torch.no_grad():
        for x, _, _ in loader(cfg, "calibration"):
            prepared(x)
    model = convert(prepared)
    meta.update(calibration_paths=[r["path"] for r in read_csv(ROOT / "data/splits/calibration.csv")],
                ptq_seconds=time.perf_counter() - started)
    export_model(model, cfg, args.model, "ptq", meta)


if __name__ == "__main__":
    main()
