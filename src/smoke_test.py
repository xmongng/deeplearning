"""CP0: test all architectures without pretrained downloads or test-set access."""
import argparse
import copy
import time
import traceback

import torch

from .common import ROOT, config, environment, seed_all, write_json
from .export_edge import export_model
from .models import NAMES, build_model
from .quantization import convert, example, prepare


def smoke_one(cfg, name):
    seed_all(cfg["seed"])
    torch.set_num_threads(4)
    model = build_model(name, len(cfg["classes"]), pretrained=False).eval()
    base = {"description": "Random weights and synthetic images; feasibility only, not experimental results"}
    export_model(copy.deepcopy(model), cfg, name, "fp32", base, smoke=True)
    ptq = prepare(copy.deepcopy(model), cfg)
    with torch.no_grad():
        for _ in range(3):
            ptq(example(cfg))
    export_model(convert(ptq), cfg, name, "ptq", base, smoke=True)
    qat = prepare(copy.deepcopy(model), cfg, qat=True)
    optimizer = torch.optim.AdamW(qat.parameters(), lr=cfg["qat"]["learning_rate"])
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(qat(example(cfg, 2)), torch.tensor([0, 1]))
        loss.backward()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite QAT loss")
        optimizer.step()
    export_model(convert(qat), cfg, name, "qat", base, smoke=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config")
    p.add_argument("--model", choices=NAMES)
    args = p.parse_args()
    cfg = config(args.config)
    result = {"environment": environment(), "models": {}, "smoke_only": True}
    for name in ([args.model] if args.model else cfg["models"]):
        start = time.perf_counter()
        try:
            smoke_one(cfg, name)
            result["models"][name] = {"status": "passed", "seconds": time.perf_counter() - start}
        except Exception:
            result["models"][name] = {"status": "failed", "traceback": traceback.format_exc()}
            traceback.print_exc()
        write_json(ROOT / f"results/cp0/{name}.json", result["models"][name])
    # Read independent per-model files to support separate invocations.
    import json
    result["models"] = {p.stem: json.loads(p.read_text()) for p in (ROOT / "results/cp0").glob("*.json")}
    write_json(ROOT / "results/cp0_feasibility.json", result)
    if any(r["status"] != "passed" for r in result["models"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
