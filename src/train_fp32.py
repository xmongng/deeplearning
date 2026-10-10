from __future__ import annotations

import argparse
import time

import torch

from .common import (ROOT, config, device_for, environment,
                     seed_all, split_fingerprint, write_csv, write_json, sha256, rng_state)
from .data import loader, loss_function
from .models import NAMES, build_model
from .training_metrics import run_epoch
from .training_history import print_epoch, save_history, plot_history


def train(cfg, name, epochs=None, device=None, smoke=False):
    seed_all(cfg["seed"])
    t = cfg["train"]
    device = device_for(device or t["device"])
    model = build_model(name, len(cfg["classes"]), pretrained=not smoke).to(device)
    model._num_classes = len(cfg["classes"])
    train_loader = loader(cfg, "train", t["batch_size"], train=True)
    val_loader = loader(cfg, "val", t["batch_size"])
    criterion = loss_function(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    count = epochs or t["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=count)
    out = ROOT / ("results/smoke/checkpoints" if smoke else "checkpoints") / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "fp32_best.pth").exists():
        raise RuntimeError(f"Checkpoint already exists at {out}; refusing to overwrite an experiment")
    evidence = ROOT / ("results/smoke/training" if smoke else "results/training") / name / "fp32"
    write_json(evidence / "run_config.json", {"config": cfg, "effective_epochs": count,
               "device": str(device), "optimizer": "AdamW", "optimizer_settings": optimizer.defaults,
               "scheduler": "CosineAnnealingLR", "scheduler_initial_state": scheduler.state_dict(),
               "training_strategy": "full_fine_tuning", "pretrained": not smoke,
               "parameters": sum(p.numel() for p in model.parameters()),
               "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "class_weights": criterion.weight.detach().cpu().tolist() if criterion.weight is not None else None,
               "split_fingerprint": split_fingerprint(), "environment": environment(),
               "metric_notes": "Train uses augmentation and training mode; val uses deterministic preprocessing and eval mode. Loss is weighted CE aggregated by target weight."})
    (evidence / "architecture.txt").write_text(str(model) + "\n")
    log, best, stale, best_epoch = [], -1.0, 0, None
    started = time.perf_counter()
    for epoch in range(count):
        epoch_start = time.perf_counter()
        model.train()
        train_scores = run_epoch(model, train_loader, criterion, device, optimizer, 2 if smoke else None)
        model.eval()
        val_scores = run_epoch(model, val_loader, criterion, device, max_batches=2 if smoke else None)
        row = {"epoch": epoch + 1, "learning_rate": optimizer.param_groups[0]["lr"],
               "seconds": time.perf_counter() - epoch_start,
               **{f"train_{k}": v for k, v in train_scores.items()},
               **{f"val_{k}": v for k, v in val_scores.items()}}
        log.append(row)
        save_history(evidence, log)
        print_epoch(name, "fp32", row, count)
        state = {"model": name, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                 "classes": cfg["classes"], "preprocessing": cfg["preprocessing"],
                 "split_fingerprint": split_fingerprint(), "epoch": epoch + 1,
                 "validation": val_scores, "config": cfg, "smoke_only": smoke}
        if val_scores["macro_f1"] > best:
            best, stale, best_epoch = val_scores["macro_f1"], 0, epoch + 1
            torch.save(state, out / "fp32_best.pth")
        else:
            stale += 1
        scheduler.step()
        torch.save({**state, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng_state": rng_state(),
                    "loader_generator_state": train_loader.generator.get_state() if hasattr(train_loader, "generator") and train_loader.generator is not None else None}, out / "fp32_last.pth")
        write_csv(out / "history.csv", log)
        if stale >= t["patience"]:
            break
    summary = {"model": name, "device": str(device),
               "parameters": sum(p.numel() for p in model.parameters()),
               "total_seconds": time.perf_counter() - started, "best_val_macro_f1": best,
               "environment": environment(), "config": cfg, "smoke_only": smoke,
               "best_epoch": best_epoch, "completed_epochs": len(log), "requested_epochs": count,
               "stop_reason": "early_stopping" if stale >= t["patience"] else "max_epochs",
               "split_fingerprint": split_fingerprint(), "best_checkpoint": str(out / "fp32_best.pth"),
               "best_checkpoint_sha256": sha256(out / "fp32_best.pth")}
    write_json(out / "training_summary.json", summary)
    write_json(evidence / "summary.json", summary)
    plot_history(evidence)
    if not smoke:
        from .evaluate_checkpoint import evaluate_checkpoint
        evaluate_checkpoint(cfg, name)


def main():
    p = argparse.ArgumentParser(description="Fine-tune ImageNet weights; select by validation Macro-F1")
    p.add_argument("--config")
    p.add_argument("--model", choices=NAMES, required=True)
    p.add_argument("--epochs", type=int)
    p.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"])
    p.add_argument("--smoke", action="store_true", help="Two batches only, random weights; excluded from final results")
    args = p.parse_args()
    train(config(args.config), args.model, args.epochs, args.device, args.smoke)


if __name__ == "__main__":
    from .stage_runner import logged_main
    logged_main("train_fp32", main)
