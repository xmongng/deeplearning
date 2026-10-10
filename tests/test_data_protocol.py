import collections

import pytest

from src.prepare_data import assert_disjoint, assign_groups, calibration


def samples():
    rows = []
    for label in range(9):
        for i in range(30):
            rows.append({"path": f"{label}/{i}.jpg", "label": str(label), "class_name": str(label),
                         "group_id": f"{label}-group-{i//2}" if i < 6 else f"{label}-{i}",
                         "sha256": f"{label}-{i//2}" if i < 4 else f"{label}-{i}"})
    # A pHash group may legitimately cross labels; keep both in the same split.
    rows[0]["group_id"] = rows[30]["group_id"] = "mixed"
    rows[1]["group_id"] = rows[31]["group_id"] = "mixed"
    return rows


def test_group_split_is_reproducible_and_has_no_leakage():
    rows = samples()
    split = assign_groups(rows, [0.8, 0.1, 0.1], 42)
    assert split == assign_groups(rows, [0.8, 0.1, 0.1], 42)
    assert_disjoint(split)
    assert sum(map(len, split.values())) == len(rows)
    for name, rs in split.items():
        counts = collections.Counter(r["label"] for r in rs)
        assert len(counts) == 9
        target = 24 if name == "train" else 3
        assert all(abs(n - target) <= 2 for n in counts.values())


def test_leakage_is_rejected():
    row = samples()[0]
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint({"train": [row], "test": [row]})


def test_calibration_uses_train_only_and_all_classes():
    train = assign_groups(samples(), [0.8, 0.1, 0.1], 42)["train"]
    selected = calibration(train, 100, 42)
    assert len(selected) == len({r["path"] for r in selected}) == 100
    assert {r["path"] for r in selected} <= {r["path"] for r in train}
    assert len({r["label"] for r in selected}) == 9
    assert selected == calibration(train, 100, 42)


def test_calibration_rejects_impossible_counts():
    with pytest.raises(ValueError):
        calibration(samples(), 8, 42)


def test_real_manifests_are_disjoint_and_match_audit():
    from src.common import ROOT, read_csv
    if not (ROOT / "data/splits/train.csv").exists():
        pytest.skip("Dataset not prepared")
    splits = {name: read_csv(ROOT / f"data/splits/{name}.csv") for name in ("train", "val", "test")}
    assert_disjoint(splits)
    audit = {r["path"] for r in read_csv(ROOT / "data/data_audit.csv") if r["status"] == "keep"}
    assert {r["path"] for rs in splits.values() for r in rs} == audit
    subset = read_csv(ROOT / "data/splits/calibration.csv")
    assert {r["path"] for r in subset} <= {r["path"] for r in splits["train"]}
