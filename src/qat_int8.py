import argparse
import time

import torch
from torchao.quantization.pt2e import disable_observer

from .common import ROOT, config, device_for, seed_all, write_csv, write_json
from .data import loader, loss_function
from .export_edge import export_model, load_fp32
from .models import NAMES
from .quantization import convert, freeze_bn, prepare
from .train_fp32 import run_epoch


def train_qat(cfg, name, epochs=None, export_only=False, device=None):
    seed_all(cfg["seed"])
    torch.set_num_threads(4)
    model, meta = load_fp32(cfg, name)
    device = device_for(device or cfg["qat"].get("device", "auto"))
    out = ROOT / "checkpoints" / name
    if export_only:
        checkpoint = torch.load(out / "qat_best.pth", map_location="cpu", weights_only=True)
        summary = __import__("json").loads((out / "qat_summary.json").read_text())
        if checkpoint["fp32_checkpoint_sha256"] != meta["fp32_checkpoint_sha256"]:
            raise ValueError("QAT checkpoint was trained from a different FP32 checkpoint")
        prepared = prepare(model, checkpoint["config"], qat=True, device="cpu")
        prepared.load_state_dict(checkpoint["state_dict"])
        export_model(convert(prepared), cfg, name, "qat", summary)
        return
    if (out / "qat_best.pth").exists():
        raise RuntimeError("QAT checkpoint already exists; refusing to overwrite")
    prepared = prepare(model, cfg, qat=True, device=device)
    prepared._num_classes = len(cfg["classes"])
    q = cfg["qat"]
    batches = loader(cfg, "train", q["batch_size"], train=True)
    validation = loader(cfg, "val", q["batch_size"])
    criterion = loss_function(cfg, device)
    optimizer = torch.optim.AdamW(prepared.parameters(), lr=q["learning_rate"],
                                 weight_decay=cfg["train"]["weight_decay"])
    best, log = -1.0, []
    start = time.perf_counter()
    for epoch in range(epochs or q["epochs"]):
        if epoch >= q["freeze_observer_epoch"]:
            prepared.apply(disable_observer)
        if epoch >= q["freeze_bn_epoch"]:
            freeze_bn(prepared)
        # The prepared graph remains in its captured training state.
        # Validate a converted deepcopy so eval never changes its BN/dropout/observers.
        train_scores = run_epoch(prepared, batches, criterion, device, optimizer)
        # Converted W8A8 graph targets CPU/XNNPACK; evaluate it on CPU while
        # keeping the trainable fake-quant graph and optimizer state on CUDA.
        candidate = convert(prepared).cpu()
        candidate._num_classes = len(cfg["classes"])
        val_scores = run_epoch(candidate, validation, loss_function(cfg, "cpu"), "cpu")
        log.append({"epoch": epoch + 1, **{f"train_{k}": v for k, v in train_scores.items()},
                    **{f"val_int8_{k}": v for k, v in val_scores.items()}, "elapsed_seconds": time.perf_counter() - start})
        print(f"{name} QAT epoch {epoch+1}: converted val F1={val_scores['macro_f1']:.4f}", flush=True)
        if val_scores["macro_f1"] > best:
            best = val_scores["macro_f1"]
            torch.save({"state_dict": prepared.state_dict(), "epoch": epoch + 1,
                        "validation_int8": val_scores, "config": cfg, **meta}, out / "qat_best.pth")
        write_csv(out / "qat_history.csv", log)
    elapsed = time.perf_counter() - start
    best_ckpt = torch.load(out / "qat_best.pth", weights_only=True, map_location="cpu")
    prepared.load_state_dict(best_ckpt["state_dict"])
    meta.update(qat_extra_seconds=elapsed, qat_best_epoch=best_ckpt["epoch"],
                qat_validation_int8=best_ckpt["validation_int8"], qat_config=q)
    write_json(out / "qat_summary.json", meta)
    export_model(convert(prepared), cfg, name, "qat", meta)


def main():
    p = argparse.ArgumentParser(description="PT2E QAT on GPU/CPU, then export for XNNPACK CPU")
    p.add_argument("--config")
    p.add_argument("--model", choices=NAMES, required=True)
    p.add_argument("--epochs", type=int)
    p.add_argument("--export-only", action="store_true", help="Retry export from the saved best QAT checkpoint without retraining")
    p.add_argument("--device", choices=["cpu", "cuda", "auto"])
    args = p.parse_args()
    train_qat(config(args.config), args.model, args.epochs, args.export_only, args.device)


if __name__ == "__main__":
    main()
