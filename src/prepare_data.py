"""Freeze stratified, duplicate-group-aware splits and train-only calibration."""
from __future__ import annotations

import argparse
import collections
import json
import random

import numpy as np

from .common import ROOT, config, read_csv, write_csv, write_json, split_fingerprint, sha256

FIELDS = ["path", "label", "class_name", "group_id", "sha256"]


def assign_groups(rows, ratios, seed):
    if len(ratios) != 3 or abs(sum(ratios) - 1) > 1e-8 or min(ratios) <= 0:
        raise ValueError("Expected three positive split ratios summing to 1")
    rng = random.Random(seed)
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["group_id"]].append(r)
    labels = sorted({r["label"] for r in rows}, key=int)
    index = {label: i for i, label in enumerate(labels)}
    totals = np.array([sum(r["label"] == label for r in rows) for label in labels])
    targets = np.array(ratios)[:, None] * totals[None, :]
    counts = np.zeros_like(targets)
    gs = list(groups.values())
    rng.shuffle(gs)
    gs.sort(key=len, reverse=True)
    splits = {s: [] for s in ("train", "val", "test")}
    for g in gs:
        vector = np.zeros(len(labels))
        for row in g:
            vector[index[row["label"]]] += 1
        # Groups may contain different labels: pHash similarity is not proof of mislabeling.
        costs = []
        for i in range(3):
            next_counts = counts.copy()
            next_counts[i] += vector
            costs.append(float(np.square(next_counts - targets).sum()))
        chosen = int(np.argmin(costs))
        counts[chosen] += vector
        splits[("train", "val", "test")[chosen]].extend(g)
    for name, rs in splits.items():
        if {r["label"] for r in rs} != set(labels):
            raise ValueError(f"Split {name} missing a class; group sizes prevent requested ratios")
        rs.sort(key=lambda r: r["path"])
    assert_disjoint(splits)
    return splits


def assert_disjoint(splits):
    for field in ("path", "group_id", "sha256"):
        seen = set()
        for rows in splits.values():
            values = {r[field] for r in rows}
            if seen & values:
                raise ValueError(f"Split leakage in {field}")
            seen |= values


def calibration(rows, count, seed):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[int(r["label"])].append(r)
    if count < len(groups) or count > len(rows):
        raise ValueError("Calibration count must cover all classes and fit train set")
    rng = random.Random(seed)
    for rs in groups.values():
        rng.shuffle(rs)
    selected = []
    # Balanced round-robin, without sampling an image more than once.
    while len(selected) < count:
        for label in sorted(groups):
            if groups[label] and len(selected) < count:
                selected.append(groups[label].pop())
    return sorted(selected, key=lambda r: r["path"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config")
    p.add_argument("--accept-conservative-groups", action="store_true",
                   help="Keep all candidate pairs in one split, including pairs not visually confirmed")
    p.add_argument("--review-complete", action="store_true", help="Record that YOU completed the manual image/label review")
    args = p.parse_args()
    cfg = config(args.config)
    out = ROOT / "data/splits"
    if (out / "train.csv").exists():
        raise RuntimeError("Splits already frozen; refusing to overwrite")
    summary = json.loads((ROOT / "data/audit_summary.json").read_text())
    pairs = read_csv(ROOT / "data/near_duplicates.csv")
    if any(r["exact"] == "True" and r["cross_label"] == "True" for r in pairs):
        raise ValueError("Identical images have conflicting labels; resolve them in review_decisions.csv first")
    if summary["candidate_pairs"] and not (args.accept_conservative_groups or args.review_complete):
        raise ValueError("Review candidate pairs or explicitly accept conservative groups")
    rows = [r for r in read_csv(ROOT / "data/data_audit.csv") if r["status"] == "keep"]
    rows = [{k: r[k] for k in FIELDS} for r in rows]
    splits = assign_groups(rows, cfg["data"]["ratios"], cfg["seed"])
    splits["calibration"] = calibration(splits["train"], cfg["data"]["calibration_images"], cfg["seed"])
    for name, rs in splits.items():
        write_csv(out / f"{name}.csv", rs, FIELDS)
    info = {"fingerprint": split_fingerprint(), "seed": cfg["seed"], "counts": {
        s: dict(collections.Counter(r["class_name"] for r in rs)) for s, rs in splits.items()},
        "manual_review_complete": args.review_complete,
        "conservative_groups_accepted": args.accept_conservative_groups,
        "audit_sha256": sha256(ROOT / "data/data_audit.csv"),
        "cross_label_phash_groups": summary["conflicting_groups"]}
    write_json(out / "summary.json", info)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
