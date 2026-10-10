"""Generate result tables, figures and a report scaffold using measured results only."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import ROOT, config
from .evaluate import verify_artifact


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument('--quality-only', action='store_true', help='Report classification/storage without requiring Mac benchmarks')
    args = p.parse_args()
    cfg = config(args.config)
    quality = pd.read_csv(ROOT / "results/quality_metrics.csv")
    if args.quality_only:
        from .quality_report import quality_report
        quality_report(cfg, quality)
        return
    bench = pd.read_csv(ROOT / "results/mac_m3_benchmark.csv")
    for _, row in quality.iterrows():
        _, meta = verify_artifact(cfg, row["model"], row["method"])
        if row["pte_sha256"] != meta["pte_sha256"]:
            raise ValueError("Quality metrics belong to different artifacts")
    selected = bench[bench.threads == args.threads]
    if len(selected) != 9:
        raise ValueError("Expected all 9 configurations at the selected thread count")
    merged = quality.merge(selected, on=["model", "method", "pte_sha256"], suffixes=("", "_benchmark"), validate="one_to_one")
    if len(merged) != 9:
        raise ValueError("Quality and benchmark artifact hashes differ")
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    colors = {"fp32": "#4477aa", "ptq": "#ee6677", "qat": "#228833"}
    for metric, filename, ylabel in (("serialized_size_mb", "model_size_comparison.png", "Serialized .pte size (MB)"),
                                     ("median_ms", "latency_comparison.png", f"Median CPU latency (ms), {args.threads} thread(s)")):
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(cfg["models"]))
        for i, method in enumerate(colors):
            values = [merged[(merged.model == n) & (merged.method == method)][metric].iloc[0] for n in cfg["models"]]
            ax.bar(x + (i - 1) * 0.25, values, 0.25, label=method.upper(), color=colors[method])
        ax.set_xticks(x, cfg["models"])
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, r in merged.iterrows():
        ax.scatter(r.median_ms, r.accuracy_pct, color=colors[r.method])
        ax.annotate(f"{r.model}/{r.method}", (r.median_ms, r.accuracy_pct), xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax.set(xlabel=f"Median CPU latency (ms), {args.threads} thread(s)", ylabel="Test accuracy (%)")
    fig.tight_layout()
    fig.savefig(figures / "accuracy_vs_latency.png", dpi=180)
    plt.close(fig)
    cm_dir = figures / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)
    for name in cfg["models"]:
        for method in colors:
            cm = json.loads((ROOT / f"results/quality/{name}/{method}_confusion.json").read_text())
            matrix = np.array(cm["matrix"])
            fig, ax = plt.subplots(figsize=(10, 9))
            ax.imshow(matrix, cmap="Blues")
            for y in range(len(matrix)):
                for x in range(len(matrix)):
                    ax.text(x, y, str(matrix[y, x]), ha="center", va="center", color="white" if matrix[y, x] > matrix.max()/2 else "black")
            ax.set_xticks(range(len(cfg["classes"])), cfg["classes"], rotation=60, ha="right")
            ax.set_yticks(range(len(cfg["classes"])), cfg["classes"])
            ax.set(xlabel="Predicted", ylabel="True", title=f"{name} / {method.upper()}")
            fig.tight_layout()
            fig.savefig(cm_dir / f"{name}_{method}.png", dpi=150)
            plt.close(fig)
    cols = ["model", "method", "accuracy_pct", "macro_f1", "delta_accuracy_pp", "serialized_size_mb",
            "compression_ratio", "median_ms", "p95_ms", "speedup", "peak_process_rss_mb", "qat_extra_seconds"]
    merged[cols].to_csv(ROOT / "results/summary.csv", index=False)
    text = ["# Experimental evaluation — RealWaste", "", "## Measured results", "",
            f"CPU backend: ExecuTorch/XNNPACK; threads: {args.threads}; all models use the same frozen test split.", "",
            merged[cols].to_markdown(index=False, floatfmt=".4f"), "",
            "Memory is lifetime process Peak RSS, including Python, PyTorch and runtime imports/load; not tensor-only memory.",
            "See raw logs for sampled inference RSS, load time, library versions and machine conditions.", "",
            "## Figures", "", "![Size](../figures/model_size_comparison.png)",
            "![Latency](../figures/latency_comparison.png)", "![Trade-off](../figures/accuracy_vs_latency.png)", "",
            "## Analysis to complete from observations", "",
            "- RQ1: Compare Accuracy, Macro-F1 and per-class scores for PTQ/QAT versus each FP32 baseline.",
            "- RQ2: Explain compression, latency and RSS differences using delegation/operator logs and repeated runs.",
            "- RQ3: Choose a trade-off based on an explicit accuracy/latency/storage requirement; consider the Pareto frontier.",
            "- RQ4: Only answer if actual Android experiments exist; otherwise state not evaluated.", "",
            "## Limitations", "", "Single seed, one principal device, class imbalance, finite test sample, pHash detection limits,",
            "manual label review status (see split summary), operator fallback, process-memory overhead,",
            "and extra QAT epochs confounding accuracy recovery. Do not claim causal explanations without controls."]
    text += ["", "## Training and stage evidence", "",
             "Epoch training metrics use augmentation/training mode; validation uses deterministic evaluation.",
             "QAT validation is measured after INT8 conversion on CPU. PTQ has calibration logs, not training epochs.",
             "Use curves and deterministic best-FP32 train/validation metrics to discuss overfitting; do not infer it from one epoch.",
             "Full commands, configs, environment, source/output hashes and failures: `results/evidence/`.", ""]
    for name in cfg["models"]:
        for phase in ("fp32", "qat"):
            history = ROOT / "results/training" / name / phase
            if (history / "epochs.csv").exists():
                text += [f"### {name} / {phase.upper()}", "",
                         f"[Epoch metrics](../results/training/{name}/{phase}/epochs.csv)",
                         f"[Run config](../results/training/{name}/{phase}/run_config.json)",
                         f"![Training curves](../results/training/{name}/{phase}/training_curves.png)", ""]
        if (ROOT / "results/baseline_validation" / name / "metrics.json").exists():
            text += [f"[Best FP32 deterministic train/validation metrics: {name}](../results/baseline_validation/{name}/metrics.json)", ""]
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs/REPORT.md").write_text("\n".join(text) + "\n")
    print("Created docs/REPORT.md, results/summary.csv and figures")


if __name__ == "__main__":
    from .stage_runner import logged_main
    logged_main("report", main)
