"""Assess the best FP32 checkpoint before quantization on train/validation only."""
import argparse
import time

import torch
from sklearn.metrics import classification_report, confusion_matrix

from .common import ROOT, check_checkpoint, checkpoint_path, config, sha256, write_csv, write_json
from .data import loader, loss_function
from .models import NAMES, build_model
from .confusion_plot import plot_confusion
from .training_metrics import run_epoch


def evaluate_checkpoint(cfg, name):
    started = time.perf_counter()
    path = checkpoint_path(name)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    check_checkpoint(checkpoint, cfg)
    if checkpoint.get('smoke_only'):
        raise ValueError('Smoke checkpoint is not an experimental baseline')
    model = build_model(name, len(cfg['classes']), pretrained=False).eval()
    model.load_state_dict(checkpoint['state_dict'])
    model._num_classes = len(cfg['classes'])
    criterion = loss_function(cfg, 'cpu')
    out = ROOT / 'results/baseline_validation' / name
    metrics = {}
    for split in ('train', 'val'):
        # Both passes use deterministic eval preprocessing, allowing a fairer gap comparison.
        batches = loader(cfg, split, cfg['train']['batch_size'], train=False)
        rows = []
        metrics[split] = run_epoch(model, batches, criterion, 'cpu', prediction_rows=rows)
        for row in rows:
            row['target_class'] = cfg['classes'][row['target']]
            row['predicted_class'] = cfg['classes'][row['prediction']]
        targets, predictions = [r['target'] for r in rows], [r['prediction'] for r in rows]
        labels = list(range(len(cfg['classes'])))
        write_csv(out / f'{split}_predictions.csv', rows)
        write_json(out / f'{split}_per_class.json', classification_report(
            targets, predictions, labels=labels, target_names=cfg['classes'], output_dict=True, zero_division=0))
        matrix = confusion_matrix(targets, predictions, labels=labels)
        write_json(out / f'{split}_confusion.json', {'classes': cfg['classes'], 'matrix': matrix.tolist()})
        plot_confusion(matrix, cfg['classes'], out / f'{split}_confusion', f'{name} FP32 / {split}')
    summary = {'model': name, 'checkpoint_epoch': checkpoint['epoch'], 'config': cfg,
               'checkpoint_sha256': sha256(path), 'split_fingerprint': checkpoint['split_fingerprint'],
               'metrics': metrics, 'seconds': time.perf_counter() - started,
               'train_minus_val_accuracy_pp': metrics['train']['accuracy'] - metrics['val']['accuracy'],
               'train_minus_val_macro_f1': metrics['train']['macro_f1'] - metrics['val']['macro_f1'],
               'notes': 'Best FP32 checkpoint; train and val use eval mode and deterministic preprocessing. Test is untouched. Gaps are evidence for analysis, not an automatic overfitting diagnosis.'}
    write_json(out / 'metrics.json', summary)
    print(f"{name} best FP32 checkpoint: train loss={metrics['train']['loss']:.5f} "
          f"F1={metrics['train']['macro_f1']:.4f}; val loss={metrics['val']['loss']:.5f} "
          f"F1={metrics['val']['macro_f1']:.4f}", flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config')
    p.add_argument('--model', required=True, choices=NAMES)
    args = p.parse_args()
    evaluate_checkpoint(config(args.config), args.model)


if __name__ == '__main__':
    from .stage_runner import logged_main
    logged_main('evaluate_checkpoint', main)
