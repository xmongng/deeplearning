"""Verify an extracted dataset or ZIP against the frozen audited manifests."""
import argparse

from .audit_data import extract
from .common import ROOT, config, read_csv, resolve, sha256


def verify(cfg):
    root = resolve(cfg["data"]["root"])
    archive_path = cfg["data"].get("archive")
    if archive_path:
        archive = resolve(archive_path)
        if not archive.is_file():
            raise FileNotFoundError(f"Dataset archive not found: {archive}")
        extract(archive, root, cfg["classes"])
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}. Set data.root to the directory containing the nine class folders.")
    missing_classes = [name for name in cfg["classes"] if not (root / name).is_dir()]
    if missing_classes:
        raise ValueError(f"Missing class folders under {root}: {missing_classes}")
    audit = read_csv(ROOT / "data/data_audit.csv")
    expected = {row["path"]: row["sha256"] for row in audit if row["status"] == "keep"}
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    actual = {path.relative_to(root).as_posix(): sha256(path)
              for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed = sorted(path for path in expected.keys() & actual.keys() if expected[path] != actual[path])
    if missing or extra or changed:
        raise RuntimeError(f"Dataset differs from audited data: missing={missing[:5]}, extra={extra[:5]}, changed={changed[:5]}")
    print(f"Verified {len(actual)} RealWaste images against data/data_audit.csv (root: {root})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    args = parser.parse_args()
    verify(config(args.config))


if __name__ == "__main__":
    from .stage_runner import logged_main
    logged_main("verify_data", main)
