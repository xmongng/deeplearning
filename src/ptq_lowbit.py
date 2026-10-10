"""Evaluate packed W4A32/W2A32 Conv/Linear weights; FP32 compute, no INT4 speed claim."""
import argparse
import json
import math
import time
import torch
from sklearn.metrics import classification_report, confusion_matrix
from .common import ROOT, config, seed_all, sha256, write_json, write_csv, environment
from .lowbit_weights import pack_model, restore_model
from .training_metrics import run_epoch


def experiment(cfg, name, bits=4, device='cpu'):
    from .export_edge import load_fp32
    from .data import loader, loss_function
    from .confusion_plot import plot_confusion
    seed_all(cfg['seed'])
    model, meta = load_fp32(cfg, name)
    out = ROOT / 'results/lowbit' / name / f'w{bits}a32'
    path = ROOT / 'exported' / name / f'w{bits}a32_packed.pth'
    metadata_path = path.with_suffix('.json')
    started = time.perf_counter()
    if path.exists():
        payload = torch.load(path, weights_only=True, map_location='cpu')
        if any(payload['metadata'][key] != meta[key] for key in ('fp32_checkpoint_sha256', 'split_fingerprint')):
            raise ValueError('Packed checkpoint/source mismatch')
        if payload['classes'] != cfg['classes'] or payload['preprocessing'] != cfg['preprocessing']:
            raise ValueError('Packed checkpoint/config mismatch')
        if metadata_path.exists() and json.loads(metadata_path.read_text())['sha256'] != sha256(path):
            raise ValueError('Packed checkpoint checksum mismatch')
    else:
        payload = pack_model(model, bits)
        payload.update(metadata=meta, classes=cfg['classes'], preprocessing=cfg['preprocessing'])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)
    if payload['bits'] != bits:
        raise ValueError('Packed bit width mismatch')
    restore_model(model, payload)
    model._num_classes = len(cfg['classes'])
    model.to(device)
    metadata = {**meta, 'bits': bits, 'scheme': f'W{bits}A32 per-output-channel symmetric weight-only',
                'inference': 'Unpack once to FP32; Conv/Linear run FP32. Not native low-bit kernels.',
                'native_lowbit_runtime': False, 'sha256': sha256(path),
                'serialized_size_mb': path.stat().st_size / 1e6,
                'quantized_layers': len(payload['packed_weights']),
                'quantized_weight_values': sum(math.prod(v['shape']) for v in payload['packed_weights'].values()),
                'config': cfg, 'device': device, 'environment': environment()}
    write_json(metadata_path, metadata)
    results = []
    for split in ('val', 'test'):
        predictions = []
        scores = run_epoch(model, loader(cfg, split), loss_function(cfg, device), device,
                           prediction_rows=predictions)
        targets = [r['target'] for r in predictions]
        labels = [r['prediction'] for r in predictions]
        report = classification_report(targets, labels, labels=list(range(len(cfg['classes']))),
                                       target_names=cfg['classes'], output_dict=True, zero_division=0)
        matrix = confusion_matrix(targets, labels, labels=list(range(len(cfg['classes']))))
        write_csv(out / f'{split}_predictions.csv', predictions)
        write_json(out / f'{split}_per_class.json', report)
        write_json(out / f'{split}_confusion.json', {'classes': cfg['classes'], 'matrix': matrix.tolist()})
        plot_confusion(matrix, cfg['classes'], out / f'{split}_confusion', f'{name} W{bits}A32 / {split}')
        row = {'model': name, 'method': f'w{bits}a32', 'split': split, **scores,
               'macro_precision': report['macro avg']['precision'], 'macro_recall': report['macro avg']['recall'],
               'weighted_f1': report['weighted avg']['f1-score'], 'images': len(predictions),
               'serialized_size_mb': metadata['serialized_size_mb'], 'packed_sha256': metadata['sha256'],
               'native_lowbit_runtime': False}
        results.append(row)
        write_json(out / f'{split}_metrics.json', {**row, 'artifact_metadata': metadata})
        print(f"{name} W{bits}A32 {split}: loss={scores['loss']:.5f}, acc={scores['accuracy']:.2f}%, F1={scores['macro_f1']:.4f}; packed={metadata['serialized_size_mb']:.3f} MB", flush=True)
    write_csv(out / 'metrics.csv', results)
    write_json(out / 'summary.json', {**metadata, 'seconds': time.perf_counter() - started, 'metrics': results})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config')
    p.add_argument('--model', required=True, choices=['mobilenet_v2', 'mobilenet_v3_large', 'shufflenet_v2_x1_0'])
    p.add_argument('--bits', type=int, choices=[2, 4], default=4)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = p.parse_args()
    experiment(config(args.config), args.model, args.bits, args.device)


if __name__ == '__main__':
    from .stage_runner import logged_main
    logged_main('ptq_lowbit', main)
