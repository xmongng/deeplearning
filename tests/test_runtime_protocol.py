import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from src.common import ROOT, config
from src.data import loader
from src.export_edge import load_runtime


def test_eval_preprocessing_is_deterministic():
    if not (ROOT / "data/splits/val.csv").exists():
        pytest.skip("Dataset not prepared")
    ds = loader(config(), "val").dataset
    a, label_a, path_a = ds[0]
    b, label_b, path_b = ds[0]
    assert a.shape == (3, 224, 224)
    assert torch.equal(a, b)
    assert (label_a, path_a) == (label_b, path_b)


@pytest.mark.parametrize("threads", [1, 4])
def test_native_executorch_thread_count(threads):
    code = f"from src.benchmark import set_threads; assert set_threads({threads}) == {threads}"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


@pytest.mark.parametrize("name", ["mobilenet_v2", "mobilenet_v3_large", "shufflenet_v2_x1_0"])
@pytest.mark.parametrize("method", ["fp32", "ptq", "qat"])
def test_smoke_artifact_reloads_with_finite_nine_class_output(name, method):
    path = ROOT / f"results/smoke/exported/{name}/{method}.pte"
    if not path.exists():
        pytest.skip("Run CP0 first")
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["smoke_only"]
    assert meta["delegates"]
    if method != "fp32":
        assert any(d["quantization_ops"] > 0 for d in meta["delegates"])
    program, forward = load_runtime(path)
    output = forward.execute((torch.zeros(1, 3, 224, 224),))[0]
    assert output.shape == (1, 9)
    assert torch.isfinite(output).all()
