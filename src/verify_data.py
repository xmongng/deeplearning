"""Extract the archive and verify it exactly matches the frozen audited manifests."""
import argparse

from .audit_data import extract
from .common import ROOT, config, read_csv, resolve, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    args = parser.parse_args()
    cfg = config(args.config)
    archive = resolve(cfg["data"]["archive"])
    root = resolve(cfg["data"]["root"])
    if not archive.exists():
        raise FileNotFoundError(f"Dataset archive not found: {archive}")
    extract(archive, root, cfg["classes"])
    audit = read_csv(ROOT / "data/data_audit.csv")
    expected = {row["path"]: row["sha256"] for row in audit if row["status"] == "keep"}
    actual = {path.relative_to(root).as_posix(): sha256(path) for path in root.rglob("*") if path.is_file()}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed = sorted(path for path in expected.keys() & actual.keys() if expected[path] != actual[path])
    if missing or extra or changed:
        raise RuntimeError(f"Dataset differs from audited data: missing={missing[:5]}, extra={extra[:5]}, changed={changed[:5]}")
    print(f"Verified {len(actual)} RealWaste images against data/data_audit.csv")


if __name__ == "__main__":
    main()
