from __future__ import annotations

import argparse
import collections
import json
import time
import shutil
from datetime import datetime, timezone

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.utils.configs import get_transform_passes
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime

from .common import (ROOT, artifact_dir, checkpoint_path, check_checkpoint, config,
                     sha256, write_json, environment, read_csv)
from .models import NAMES, build_model
from .quantization import example
from .export_validation import compare_outputs, summarize_parity


def load_runtime(path):
    runtime = Runtime.get()
    if not runtime.backend_registry.is_available("XnnpackBackend"):
        raise RuntimeError("This ExecuTorch runtime was built without XNNPACK")
    program = runtime.load_program(path)
    return program, program.load_method("forward")


def export_model(model, cfg, name, method, metadata, smoke=False):
    started = time.perf_counter()
    out = artifact_dir(name, smoke)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    evidence = ROOT / "results/export_validation" / name / method / stamp
    evidence.mkdir(parents=True, exist_ok=False)
    report = {**metadata, "model": name, "method": method, "classes": cfg["classes"],
              "preprocessing": cfg["preprocessing"], "backend": "XNNPACK", "smoke_only": smoke,
              "config": cfg, "export_status": "running", "evidence_directory": str(evidence.relative_to(ROOT))}
    write_json(evidence / "export_report.json", report)
    promoted = []
    try:
        final_path = out / f"{method}.pte"
        final_meta = out / f"{method}.json"
        if final_path.exists() and final_meta.exists():
            raise RuntimeError(f"Artifact already exists: {final_path}")
        # Old versions wrote the .pte before validation and could leave an orphan.
        legacy = [out / f"{method}{suffix}" for suffix in (".pte", ".json", ".pt2", "_edge_graph.txt")]
        existing = [path for path in legacy if path.exists()]
        if existing:
            recovery = evidence / "legacy_incomplete"
            recovery.mkdir()
            hashes = {path.name: sha256(path) for path in existing}
            for path in existing:
                shutil.move(str(path), recovery / path.name)
            write_json(recovery / "recovery.json", {"reason": "Incomplete artifact pair from an earlier export",
                       "original_directory": str(out), "sha256": hashes})
            print(f"Preserved incomplete earlier export at {recovery}", flush=True)
        report["environment"] = environment()
        model = model.cpu()
        x = example(cfg)
        with torch.no_grad():
            reference = model(x).detach()
        ep = torch.export.export(model, (x,))
        ops = collections.Counter(str(n.target) for n in ep.graph.nodes if n.op == "call_function")
        qops = {k: v for k, v in ops.items() if "quantized_decomposed" in k}
        report.update(graph_operators=dict(ops), quantization_operators=qops)
        if method != "fp32" and not qops:
            raise RuntimeError("INT8 branch contains no quantize/dequantize operators")
        torch.export.save(ep, evidence / f"{method}.pt2")
        lowered = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()],
                                            transform_passes=get_transform_passes())
        edge_module = lowered.exported_program().graph_module
        (evidence / f"{method}_edge_graph.txt").write_text(str(edge_module))
        delegates = []
        for module in edge_module.modules():
            if hasattr(module, "backend_id") and hasattr(module, "original_module"):
                subops = collections.Counter(str(n.target) for n in module.original_module.graph.nodes if n.op == "call_function")
                delegates.append({"backend": module.backend_id, "operators": dict(subops),
                                  "quantization_ops": sum(v for k, v in subops.items() if "quantized_decomposed" in k)})
        report["delegates"] = delegates
        if not delegates:
            raise RuntimeError("No XNNPACK delegate produced; cannot claim an XNNPACK benchmark")
        if method != "fp32" and not any(d["quantization_ops"] for d in delegates):
            raise RuntimeError("Quantization graph not delegated to XNNPACK")
        residual = collections.Counter(str(n.target) for n in edge_module.graph.nodes if n.op == "call_function")
        report["remaining_edge_operators"] = dict(residual)
        path = evidence / f"{method}.pte"
        path.write_bytes(lowered.to_executorch().buffer)
        report.update(pte_sha256=sha256(path), serialized_size_mb=path.stat().st_size / 1e6)
        loaded, forward = load_runtime(path)
        rows = []
        with torch.no_grad():
            actual = forward.execute((x,))[0].clone()
            rows.append(compare_outputs(reference, actual, "synthetic", cfg["classes"], method))
        write_json(evidence / "parity_samples.json", rows)
        if not smoke:
            from .data import loader
            with torch.no_grad():
                # The complete validation set is used for INT8 prediction parity, never test.
                for sample, target, paths in loader(cfg, "val"):
                    expected = model(sample).detach()
                    observed = forward.execute((sample,))[0].clone()
                    rows.append(compare_outputs(expected, observed, paths[0], cfg["classes"], method, target.item()))
                    write_json(evidence / "parity_samples.json", rows)
        expected_samples = len(read_csv(ROOT / 'data/splits/val.csv')) if not smoke and (ROOT / 'data/splits/val.csv').exists() else None
        if expected_samples is not None and len(rows) - 1 != expected_samples:
            raise RuntimeError('Incomplete validation coverage')
        parity = summarize_parity(rows, method, len(cfg["classes"]), smoke,
                                  cfg.get('export_validation', {}).get('int8_policy', 'exact_validation_predictions'))
        report.update(parity=parity, parity_status=parity["status"],
                      parity_max_abs_error=parity["max_abs_error"], parity_samples=len(rows),
                      parity_argmax_agreement=parity["argmax_agreement"])
        write_json(evidence / "parity_summary.json", parity)
        for warning in parity['warnings']:
            print('WARNING: ' + warning, flush=True)
        if parity["status"] != "passed":
            raise RuntimeError(f"Export parity failed: {parity['failure_reasons']}. Diagnostics: {evidence}")
        report.update(export_status="passed", export_seconds=time.perf_counter() - started)
        write_json(evidence / "export_report.json", report)
        # Publish metadata last: its presence commits an already validated artifact pair.
        for suffix in (".pt2", "_edge_graph.txt", ".pte"):
            destination = out / f"{method}{suffix}"
            promoted.append(destination)
            shutil.copy2(evidence / f"{method}{suffix}", destination)
        write_json(final_meta, report)
        promoted.append(final_meta)
        print(f"Exported {final_path}: {report['serialized_size_mb']:.2f} MB, {len(delegates)} delegates; "
              f"parity={parity['policy']}, numerical mismatch samples={parity['numerical_mismatch_samples']}", flush=True)
        return report
    except BaseException as exc:
        # Preserve diagnostics even for parity, lowering, runtime or publication failures.
        report.update(export_status="failed", export_seconds=time.perf_counter() - started,
                      error_type=type(exc).__name__, error=str(exc))
        for path in promoted:
            if path.exists():
                shutil.move(str(path), evidence / ("unpublished_" + path.name))
        write_json(evidence / "export_report.json", report)
        raise


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
