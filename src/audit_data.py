"""Extract safely, validate images, detect duplicates before splitting."""
from __future__ import annotations

import argparse
import collections
import html
import io
import json
import zipfile
from pathlib import Path

import imagehash
from PIL import Image, ImageOps

from .common import ROOT, config, read_csv, resolve, sha256, write_csv, write_json


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        x, y = self.find(x), self.find(y)
        self.parent[max(x, y)] = min(x, y)


def extract(archive, root, classes):
    root.mkdir(parents=True, exist_ok=True)
    seen = set()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            p = Path(info.filename)
            if info.is_dir() or "__MACOSX" in p.parts or p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            if p.parent.name not in classes:
                raise ValueError(f"Unknown image class in ZIP: {p}")
            if ".." in p.parts or p.is_absolute():
                raise ValueError(f"Unsafe ZIP entry: {p}")
            dest = root / p.parent.name / p.name
            if dest in seen:
                raise ValueError(f"Repeated destination in ZIP: {dest}")
            seen.add(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            content = z.read(info)
            if dest.exists() and dest.read_bytes() != content:
                raise ValueError(f"Existing raw image differs from ZIP: {dest}")
            if not dest.exists():
                dest.write_bytes(content)
    if not seen:
        raise ValueError("Archive contains no supported images")


def audit(cfg):
    if any((ROOT / "checkpoints").rglob("*.pth")):
        raise RuntimeError("Checkpoints exist. Audit/split changes require a new project copy or deliberate archival of the previous run.")
    root = resolve(cfg["data"]["root"])
    archive = resolve(cfg["data"]["archive"])
    extract(archive, root, cfg["classes"])
    decisions_file = ROOT / "data/review_decisions.csv"
    decisions = {r["path"]: r for r in read_csv(decisions_file)} if decisions_file.exists() else {}
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        rel = path.relative_to(root).as_posix()
        cls = path.parent.name
        row = dict(path=rel, original_class=cls, class_name=cls, label=cfg["classes"].index(cls),
                   status="keep", reason="", sha256=sha256(path), pixel_sha256="", phash="", width="", height="", group_id="")
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.load()
                row.update(width=im.width, height=im.height,
                           pixel_sha256=__import__("hashlib").sha256(f"{im.size}".encode() + im.tobytes()).hexdigest(),
                           phash=str(imagehash.phash(im)))
        except (OSError, ValueError, SyntaxError) as e:
            row.update(status="exclude", reason=f"unreadable: {e}")
        if rel in decisions:
            d = decisions[rel]
            if not d["reason"].strip():
                raise ValueError(f"Review decision needs a reason: {rel}")
            if d["action"] == "exclude":
                row.update(status="exclude", reason=d["reason"])
            elif d["action"] == "relabel" and row["status"] == "keep":
                cls = d["new_class"]
                row.update(class_name=cls, label=cfg["classes"].index(cls), reason=d["reason"])
            else:
                raise ValueError(f"Invalid review decision: {d}")
        rows.append(row)
    unknown = set(decisions) - {r["path"] for r in rows}
    if unknown:
        raise ValueError(f"Review decisions reference unknown paths: {unknown}")
    valid = [r for r in rows if r["status"] == "keep"]
    uf = UnionFind(len(valid))
    pairs = []
    hashes = [int(r["phash"], 16) for r in valid]
    threshold = cfg["data"]["phash_distance"]
    # ~11 million 64-bit comparisons for 4,752 images; no quadratic image storage.
    for i, left in enumerate(valid):
        for j in range(i + 1, len(valid)):
            distance = (hashes[i] ^ hashes[j]).bit_count()
            exact = left["sha256"] == valid[j]["sha256"] or left["pixel_sha256"] == valid[j]["pixel_sha256"]
            if exact or distance <= threshold:
                uf.union(i, j)
                pairs.append(dict(path_a=left["path"], path_b=valid[j]["path"],
                                  distance=distance, exact=exact,
                                  cross_label=left["label"] != valid[j]["label"]))
    groups = collections.defaultdict(list)
    for i, row in enumerate(valid):
        row["group_id"] = f"g{uf.find(i):05d}"
        groups[row["group_id"]].append(row)
    conflicts = [g for g, rs in groups.items() if len({r["label"] for r in rs}) > 1]
    write_csv(ROOT / "data/data_audit.csv", rows)
    write_csv(ROOT / "data/near_duplicates.csv", pairs,
              ["path_a", "path_b", "distance", "exact", "cross_label"])
    summary = {"archive_sha256": sha256(archive), "images": len(rows), "usable": len(valid),
               "excluded": len(rows) - len(valid), "candidate_pairs": len(pairs),
               "groups": len(groups), "conflicting_groups": conflicts,
               "phash_distance": threshold, "manual_label_review_complete": False,
               "before": dict(collections.Counter(r["original_class"] for r in rows)),
               "after": dict(collections.Counter(r["class_name"] for r in valid))}
    write_json(ROOT / "data/audit_summary.json", summary)
    report = ["# RealWaste — Data audit", "", f"Archive SHA-256: `{summary['archive_sha256']}`", "",
              f"Images: {len(rows)}; readable and retained: {len(valid)}; excluded: {summary['excluded']}.",
              f"Candidate duplicate pairs: {len(pairs)}; cross-label groups: {len(conflicts)}.", "",
              "| Class | Before | After |", "|---|---:|---:|"]
    report += [f"| {c} | {summary['before'].get(c, 0)} | {summary['after'].get(c, 0)} |" for c in cfg["classes"]]
    report += ["", "Exact and near-duplicate images are retained but grouped in the same split.",
               "pHash candidates are conservative leakage groups, not verified duplicate labels.",
               "Review data/review.html and suspicious labels before freezing splits.",
               "Record exclusions/relabels in data/review_decisions.csv and rerun audit.",
               "Automatic checks cannot prove that all labels are correct or all near duplicates were found."]
    (ROOT / "docs/DATA_AUDIT.md").write_text("\n".join(report) + "\n")
    cards = []
    for pair in pairs:
        cards.append("<section>" + html.escape(str(pair)) + "<br>" + "".join(
            f'<a href="raw/{html.escape(pair[k], quote=True)}"><img loading="lazy" width="224" src="raw/{html.escape(pair[k], quote=True)}"></a>'
            for k in ("path_a", "path_b")) + "</section>")
    (ROOT / "data/review.html").write_text('<meta charset="utf-8"><title>Candidate duplicate review</title><h1>Review candidate pairs</h1>' + "\n".join(cards))
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config")
    args = p.parse_args()
    (ROOT / "docs").mkdir(exist_ok=True)
    audit(config(args.config))


if __name__ == "__main__":
    main()
