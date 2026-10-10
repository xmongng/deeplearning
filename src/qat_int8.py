import argparse
import json
import time

import torch
from torchao.quantization.pt2e import disable_observer

from .common import (ROOT, config, device_for, seed_all, write_csv, write_json,
                     environment, split_fingerprint, sha256, rng_state, restore_rng_state)
from .data import loader, loss_function
from .export_edge import export_model, load_fp32
from .models import NAMES
from .quantization import convert, freeze_bn, prepare
from .training_metrics import run_epoch
from .training_history import print_epoch, save_history, plot_history


def train_qat(cfg, name, epochs=None, export_only=False, device=None, resume=False):
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
    if (out / "qat_best.pth").exists() and not resume:
        raise RuntimeError("QAT checkpoint already exists; refusing to overwrite")
    checkpoint = None
    if resume:
        if (out / 'qat_summary.json').exists():
            raise ValueError('QAT is complete; use --export-only')
        checkpoint = torch.load(out / 'qat_last.pth', map_location='cpu', weights_only=True)
        if checkpoint['fp32_checkpoint_sha256'] != meta['fp32_checkpoint_sha256'] or checkpoint['split_fingerprint'] != split_fingerprint():
            raise ValueError('QAT resume checkpoint/source or split mismatch')
        for key in ('seed', 'classes', 'preprocessing', 'qat', 'train'):
            if checkpoint['config'][key] != cfg[key]:
                raise ValueError('Resume configuration changed: ' + key)
        if not (out / 'qat_best.pth').exists():
            raise ValueError('Resume requires qat_best.pth as well as qat_last.pth')
    prepared = prepare(model, cfg, qat=True, device=device)
    prepared._num_classes = len(cfg["classes"])
    q = cfg["qat"]
    batches = loader(cfg, "train", q["batch_size"], train=True)
    validation = loader(cfg, "val", q["batch_size"])
    criterion = loss_function(cfg, device)
    optimizer = torch.optim.AdamW(prepared.parameters(), lr=q["learning_rate"],
                                 weight_decay=cfg["train"]["weight_decay"])
    count = epochs or q["epochs"]
    evidence = ROOT / "results/training" / name / "qat"
    first_epoch = 0
    best, log = -1.0, []
    prior_elapsed = 0.0
    if checkpoint is not None:
        first_epoch = checkpoint['epoch']
        if count < first_epoch:
            raise ValueError('Requested total epochs is smaller than saved epoch')
        prepared.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        saved_best = torch.load(out / 'qat_best.pth', map_location='cpu', weights_only=True)
        if saved_best['fp32_checkpoint_sha256'] != meta['fp32_checkpoint_sha256'] or saved_best['epoch'] > first_epoch:
            raise ValueError('Inconsistent QAT best/last checkpoints')
        best = saved_best['validation_int8']['macro_f1']
        log = json.loads((evidence / 'epochs.json').read_text())
        log = [row for row in log if row['epoch'] <= first_epoch]
        if [row['epoch'] for row in log] != list(range(1, first_epoch + 1)):
            raise ValueError('Missing or inconsistent epoch history for resume')
        prior_elapsed = log[-1]['elapsed_seconds'] if log else 0.0
        restore_rng_state(checkpoint['rng_state'])
        if checkpoint.get('loader_generator_state') is not None:
            batches.generator.set_state(checkpoint['loader_generator_state'].cpu())
        write_json(evidence / f'resume_from_epoch_{first_epoch}.json', {
            'checkpoint_sha256': sha256(out / 'qat_last.pth'), 'epoch': first_epoch,
            'config': cfg, 'environment': environment()})
        print(f'{name}: resumed QAT after epoch {first_epoch}; total epochs={count}', flush=True)
    write_json(evidence / ('resume_run_config.json' if resume else 'run_config.json'), {"config": cfg, "effective_epochs": count,
               "device": str(device), "optimizer": "AdamW", "optimizer_settings": optimizer.defaults, "scheduler": None,
               "class_weights": criterion.weight.detach().cpu().tolist() if criterion.weight is not None else None,
               "environment": environment(), "split_fingerprint": split_fingerprint(), **meta,
               "parameters": sum(p.numel() for p in prepared.parameters()),
               "trainable_parameters": sum(p.numel() for p in prepared.parameters() if p.requires_grad),
               "metric_notes": "Train uses fake quantization and augmentation; validation uses converted INT8 on CPU. Observer/BN freeze epochs below are one-based."})
    (evidence / "prepared_graph.txt").write_text(str(prepared) + "\n")
    start = time.perf_counter()
    for epoch in range(first_epoch, count):
        epoch_start = time.perf_counter()
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
        row = {"epoch": epoch + 1, "learning_rate": optimizer.param_groups[0]["lr"],
                    "seconds": time.perf_counter() - epoch_start,
                    "observer_frozen": epoch >= q["freeze_observer_epoch"],
                    "batchnorm_frozen": epoch >= q["freeze_bn_epoch"], **{f"train_{k}": v for k, v in train_scores.items()},
                    **{f"val_int8_{k}": v for k, v in val_scores.items()}, "elapsed_seconds": prior_elapsed + time.perf_counter() - start}
        log.append(row)
        save_history(evidence, log)
        print_epoch(name, "qat", row, count)
        if val_scores["macro_f1"] > best:
            best = val_scores["macro_f1"]
            torch.save({"state_dict": prepared.state_dict(), "epoch": epoch + 1,
                        "validation_int8": val_scores, "config": cfg, **meta}, out / "qat_best.pth")
        torch.save({"state_dict": prepared.state_dict(), "epoch": epoch + 1,
                    "optimizer": optimizer.state_dict(), "config": cfg, "rng_state": rng_state(),
                    "loader_generator_state": batches.generator.get_state() if hasattr(batches, "generator") and batches.generator is not None else None,
                    "validation_int8": val_scores, **meta}, out / "qat_last.pth")
        write_csv(out / "qat_history.csv", log)
    elapsed = prior_elapsed + time.perf_counter() - start
    best_ckpt = torch.load(out / "qat_best.pth", weights_only=True, map_location="cpu")
    prepared.load_state_dict(best_ckpt["state_dict"])
    meta.update(qat_extra_seconds=elapsed, qat_best_epoch=best_ckpt["epoch"],
                qat_validation_int8=best_ckpt["validation_int8"], qat_config=q,
                completed_epochs=len(log), requested_epochs=count, config=cfg, environment=environment(),
                device=str(device), best_checkpoint_sha256=sha256(out / "qat_best.pth"),
                observer_freeze_epoch_one_based=q["freeze_observer_epoch"] + 1,
                batchnorm_freeze_epoch_one_based=q["freeze_bn_epoch"] + 1)
    write_json(out / "qat_summary.json", meta)
    write_json(evidence / "summary.json", meta)
    plot_history(evidence)
    export_model(convert(prepared), cfg, name, "qat", meta)


def main():
    p = argparse.ArgumentParser(description="PT2E QAT on GPU/CPU, then export for XNNPACK CPU")
    p.add_argument("--config")
    p.add_argument("--model", choices=NAMES, required=True)
    p.add_argument("--epochs", type=int)
    p.add_argument("--export-only", action="store_true", help="Retry export from the saved best QAT checkpoint without retraining")
    p.add_argument('--resume', action='store_true', help='Continue qat_last.pth, preserving optimizer, RNG, best and epoch history')
    p.add_argument("--device", choices=["cpu", "cuda", "auto"])
    args = p.parse_args()
    if args.resume and args.export_only:
        p.error('--resume and --export-only are mutually exclusive')
    train_qat(config(args.config), args.model, args.epochs, args.export_only, args.device, args.resume)


if __name__ == "__main__":
    from .stage_runner import logged_main
    logged_main("qat_int8", main)
