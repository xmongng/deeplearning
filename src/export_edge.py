from __future__ import annotations

import argparse
import collections
import json
import time

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.utils.configs import get_transform_passes
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime

from .common import (ROOT, artifact_dir, checkpoint_path, check_checkpoint, config,
                     sha256, write_json, environment)
from .models import NAMES, build_model
from .quantization import example


def load_runtime(path):
    runtime = Runtime.get()
    if not runtime.backend_registry.is_available("XnnpackBackend"):
        raise RuntimeError("This ExecuTorch runtime was built without XNNPACK")
    program = runtime.load_program(path)
    return program, program.load_method("forward")


def export_model(model, cfg, name, method, metadata, smoke=False):
    started = time.perf_counter()
    model = model.cpu()
    out = artifact_dir(name, smoke)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{method}.pte"
    if path.exists():
        raise RuntimeError(f"Artifact already exists: {path}")
    x = example(cfg)
    with torch.no_grad():
        reference = model(x).detach()
    ep = torch.export.export(model, (x,))
    ops = collections.Counter(str(n.target) for n in ep.graph.nodes if n.op == "call_function")
    qops = {k: v for k, v in ops.items() if "quantized_decomposed" in k}
    if method != "fp32" and not qops:
        raise RuntimeError("INT8 branch contains no quantize/dequantize operators")
    torch.export.save(ep, out / f"{method}.pt2")
    lowered = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()],
                                        transform_passes=get_transform_passes())
    edge_module = lowered.exported_program().graph_module
    delegates = []
    for module in edge_module.modules():
        if hasattr(module, "backend_id") and hasattr(module, "original_module"):
            subops = collections.Counter(str(n.target) for n in module.original_module.graph.nodes if n.op == "call_function")
            delegates.append({"backend": module.backend_id, "operators": dict(subops),
                              "quantization_ops": sum(v for k, v in subops.items() if "quantized_decomposed" in k)})
    if not delegates:
        raise RuntimeError("No XNNPACK delegate produced; cannot claim an XNNPACK benchmark")
    if method != "fp32" and not any(d["quantization_ops"] for d in delegates):
        raise RuntimeError("Quantization graph not delegated to XNNPACK")
    residual = collections.Counter(str(n.target) for n in edge_module.graph.nodes if n.op == "call_function")
    program = lowered.to_executorch()
    path.write_bytes(program.buffer)
    loaded, forward = load_runtime(path)
    with torch.no_grad():
        actual = forward.execute((x,))[0].clone()
    if actual.shape != (1, len(cfg["classes"])) or not torch.isfinite(actual).all():
        raise RuntimeError("Invalid deployed output")
    torch.testing.assert_close(actual, reference, rtol=1e-3, atol=1e-3 if method == "fp32" else 0.15)
    parity_errors = [(reference - actual).abs().max().item()]
    parity_matches = [bool(reference.argmax(1).eq(actual.argmax(1)).all())]
    if not smoke:
        from .data import loader
        with torch.no_grad():
            for i, (sample, _, _) in enumerate(loader(cfg, "val")):
                if i >= 10:
                    break
                expected = model(sample).detach()
                observed = forward.execute((sample,))[0].clone()
                torch.testing.assert_close(observed, expected, rtol=1e-3, atol=1e-3 if method == "fp32" else 0.15)
                parity_errors.append((expected - observed).abs().max().item())
                parity_matches.append(bool(expected.argmax(1).eq(observed.argmax(1)).all()))
    report = {**metadata, "model": name, "method": method, "classes": cfg["classes"],
              "preprocessing": cfg["preprocessing"], "backend": "XNNPACK", "smoke_only": smoke,
              "pte_sha256": sha256(path), "serialized_size_mb": path.stat().st_size / 1e6,
              "graph_operators": dict(ops), "quantization_operators": qops,
              "delegates": delegates, "remaining_edge_operators": dict(residual),
              "parity_max_abs_error": max(parity_errors), "parity_samples": len(parity_errors),
              "parity_argmax_agreement": sum(parity_matches) / len(parity_matches),
              "export_seconds": time.perf_counter() - started, "config": cfg, "environment": environment()}
    write_json(out / f"{method}.json", report)
    (out / f"{method}_edge_graph.txt").write_text(str(edge_module))
    print(f"Exported {path}: {report['serialized_size_mb']:.2f} MB, {len(delegates)} delegates", flush=True)
    return report


def load_fp32(cfg, name):
    path = checkpoint_path(name)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    check_checkpoint(ckpt, cfg)
    if ckpt.get("smoke_only"):
        raise ValueError("Cannot use smoke checkpoint in final experiments")
    model = build_model(name, len(cfg["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, {"fp32_checkpoint_sha256": sha256(path), "split_fingerprint": ckpt["split_fingerprint"],
                   "checkpoint_epoch": ckpt["epoch"]}


def main():
    p = argparse.ArgumentParser(description="Export trained FP32 checkpoint to ExecuTorch/XNNPACK")
    p.add_argument("--config")
    p.add_argument("--model", required=True, choices=NAMES)
    args = p.parse_args()
    cfg = config(args.config)
    model, meta = load_fp32(cfg, args.model)
    export_model(model, cfg, args.model, "fp32", meta)


if __name__ == "__main__":
    from .stage_runner import logged_main
    logged_main("export_edge", main)
