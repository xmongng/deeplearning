from __future__ import annotations

import argparse
import time

import torch
from sklearn.metrics import accuracy_score, f1_score

from .common import (ROOT, config, device_for, environment,
                     seed_all, split_fingerprint, write_csv, write_json)
from .data import loader, loss_function
from .models import NAMES, build_model


def run_epoch(model, batches, criterion, device, optimizer=None, max_batches=None):
    predictions, targets = [], []
    total_loss, total = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for i, (x, y, _) in enumerate(batches):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            total += len(y)
            predictions.extend(logits.detach().argmax(1).cpu().tolist())
            targets.extend(y.cpu().tolist())
    if not total:
        raise ValueError("Empty epoch")
    return {"loss": total_loss / total, "accuracy": accuracy_score(targets, predictions) * 100,
            "macro_f1": f1_score(targets, predictions, labels=list(range(model._num_classes)) if hasattr(model, "_num_classes") else None,
                     average="macro", zero_division=0)}


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
    log, best, stale = [], -1.0, 0
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
        print(f"{name} epoch {epoch+1}/{count}: val acc={val_scores['accuracy']:.2f}% F1={val_scores['macro_f1']:.4f}", flush=True)
        state = {"model": name, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                 "classes": cfg["classes"], "preprocessing": cfg["preprocessing"],
                 "split_fingerprint": split_fingerprint(), "epoch": epoch + 1,
                 "validation": val_scores, "config": cfg, "smoke_only": smoke}
        if val_scores["macro_f1"] > best:
            best, stale = val_scores["macro_f1"], 0
            torch.save(state, out / "fp32_best.pth")
        else:
            stale += 1
        scheduler.step()
        torch.save({**state, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, out / "fp32_last.pth")
        write_csv(out / "history.csv", log)
        if stale >= t["patience"]:
            break
    write_json(out / "training_summary.json", {"model": name, "device": str(device),
               "parameters": sum(p.numel() for p in model.parameters()),
               "total_seconds": time.perf_counter() - started, "best_val_macro_f1": best,
               "environment": environment(), "config": cfg, "smoke_only": smoke})


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
    main()
