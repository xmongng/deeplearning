import zipfile

import pytest

from src import verify_data
from src.common import sha256


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    root = tmp_path / "RealWaste"
    (root / "Cardboard").mkdir(parents=True)
    image = root / "Cardboard/example.jpg"
    image.write_bytes(b"frozen-image-content")
    rows = [{"path": "Cardboard/example.jpg", "sha256": sha256(image), "status": "keep"}]
    monkeypatch.setattr(verify_data, "read_csv", lambda _: rows)
    return root, {"data": {"root": str(root), "archive": None}, "classes": ["Cardboard"]}


def test_extracted_dataset_is_read_only_and_ignores_non_images(dataset, monkeypatch):
    root, cfg = dataset
    (root / "README.md").write_text("Dataset information")
    monkeypatch.setattr(verify_data, "extract", lambda *args: pytest.fail("Must not extract directory input"))
    verify_data.verify(cfg)
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()) == ["Cardboard/example.jpg", "README.md"]


@pytest.mark.parametrize("difference", ["missing", "extra", "changed"])
def test_manifest_mismatch_is_rejected(dataset, difference):
    root, cfg = dataset
    if difference == "missing":
        (root / "Cardboard/example.jpg").unlink()
    elif difference == "extra":
        (root / "Cardboard/extra.jpg").write_bytes(b"extra")
    else:
        (root / "Cardboard/example.jpg").write_bytes(b"modified")
    with pytest.raises(RuntimeError, match="Dataset differs"):
        verify_data.verify(cfg)


def test_wrong_directory_level_is_rejected(dataset):
    root, cfg = dataset
    cfg["data"]["root"] = str(root.parent)
    with pytest.raises(ValueError, match="Missing class folders"):
        verify_data.verify(cfg)


def test_zip_source_still_extracts_and_verifies(dataset, tmp_path):
    root, cfg = dataset
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.write(root / "Cardboard/example.jpg", "realwaste-main/RealWaste/Cardboard/example.jpg")
    cfg["data"] = {"root": str(tmp_path / "raw"), "archive": str(archive)}
    verify_data.verify(cfg)
    assert (tmp_path / "raw/Cardboard/example.jpg").is_file()
