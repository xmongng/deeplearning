"""Final quality evaluation using deployed .pte files, batch size 1."""
import argparse
import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .common import ROOT, artifact_dir, config, sha256, split_fingerprint, write_csv, write_json
from .data import loader
from .export_edge import load_runtime


def verify_artifact(cfg, name, method):
    path = artifact_dir(name) / f"{method}.pte"
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta["smoke_only"] or meta["classes"] != cfg["classes"] or meta["preprocessing"] != cfg["preprocessing"]:
        raise ValueError("Artifact/config mismatch or smoke artifact")
    if meta["split_fingerprint"] != split_fingerprint() or meta["pte_sha256"] != sha256(path):
        raise ValueError("Artifact or data splits changed")
    return path, meta


def evaluate_one(cfg, name, method):
    path, meta = verify_artifact(cfg, name, method)
    program, forward = load_runtime(path)
    rows = []
    with torch.no_grad():
        for x, y, paths in loader(cfg, "test"):
            logits = forward.execute((x,))[0].clone()
            if logits.shape != (1, len(cfg["classes"])) or not torch.isfinite(logits).all():
                raise RuntimeError("Invalid runtime output")
            prediction = logits.argmax(1).item()
            rows.append({"path": paths[0], "target": y.item(), "prediction": prediction})
    targets, predicted = [r["target"] for r in rows], [r["prediction"] for r in rows]
    labels = list(range(len(cfg["classes"])))
    scores = classification_report(targets, predicted, labels=labels, target_names=cfg["classes"],
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(targets, predicted, labels=labels)
    out = ROOT / "results/quality" / name
    write_csv(out / f"{method}_predictions.csv", rows)
    write_json(out / f"{method}_per_class.json", scores)
    write_json(out / f"{method}_confusion.json", {"classes": cfg["classes"], "matrix": cm.tolist()})
    return {"model": name, "method": method, "accuracy_pct": accuracy_score(targets, predicted) * 100,
            "macro_f1": f1_score(targets, predicted, labels=labels, average="macro", zero_division=0),
            "test_images": len(rows), "serialized_size_mb": meta["serialized_size_mb"],
            "qat_extra_seconds": meta.get("qat_extra_seconds", ""), "pte_sha256": meta["pte_sha256"],
            "split_fingerprint": meta["split_fingerprint"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config")
    args = p.parse_args()
    cfg = config(args.config)
    # Preflight ALL artifacts before touching test data.
    for name in cfg["models"]:
        for method in ("fp32", "ptq", "qat"):
            verify_artifact(cfg, name, method)
    rows = []
    torch.set_num_threads(1)
    for name in cfg["models"]:
        group = [evaluate_one(cfg, name, method) for method in ("fp32", "ptq", "qat")]
        base = group[0]
        for row in group:
            row["delta_accuracy_pp"] = row["accuracy_pct"] - base["accuracy_pct"]
            row["compression_ratio"] = base["serialized_size_mb"] / row["serialized_size_mb"]
            rows.append(row)
            print(f"{name}/{row['method']}: accuracy={row['accuracy_pct']:.2f}%, Macro-F1={row['macro_f1']:.4f}", flush=True)
    write_csv(ROOT / "results/quality_metrics.csv", rows)


if __name__ == "__main__":
    main()
