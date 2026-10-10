"""Classification/storage report, keeping native INT8 and packed low-bit distinct."""
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT, sha256
from .evaluate import verify_artifact


def markdown_table(frame):
    def cell(value):
        text = f'{value:.6f}' if isinstance(value, (float, np.floating)) else str(value)
        return text.replace('|', '\\|').replace('\n', ' ')
    lines = ['| ' + ' | '.join(map(str, frame.columns)) + ' |',
             '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    lines.extend('| ' + ' | '.join(cell(v) for v in row) + ' |'
                 for row in frame.itertuples(index=False, name=None))
    return '\n'.join(lines)


def quality_report(cfg, quality):
    rows, drift = [], []
    for _, row in quality.iterrows():
        _, meta = verify_artifact(cfg, row['model'], row['method'])
        if row['pte_sha256'] != meta['pte_sha256']:
            raise ValueError('Quality/artifact hash mismatch')
        rows.append({'model': row['model'], 'method': row['method'], 'accuracy_pct': row['accuracy_pct'],
                     'macro_f1': row['macro_f1'], 'loss': row['loss'], 'serialized_size_mb': row['serialized_size_mb'],
                     'format': 'ExecuTorch .pte', 'compute': 'XNNPACK FP32' if row['method'] == 'fp32' else 'XNNPACK INT8'})
        parity = meta.get('parity', {})
        drift.append({'model': row['model'], 'method': row['method'], 'policy': parity.get('policy'),
                      'agreement': parity.get('validation_argmax_agreement'),
                      'delta_val_accuracy_pp': parity.get('runtime_minus_reference_accuracy_pp'),
                      'delta_val_macro_f1': parity.get('runtime_minus_reference_macro_f1'),
                      'max_logit_error': parity.get('max_abs_error')})
    for model in cfg['models']:
        for width in (4, 2):
            metric = ROOT / f'results/lowbit/{model}/w{width}a32/test_metrics.json'
            if not metric.exists():
                continue
            data = json.loads(metric.read_text())
            path = ROOT / f'exported/{model}/w{width}a32_packed.pth'
            if sha256(path) != data['packed_sha256']:
                raise ValueError('Low-bit metrics/artifact mismatch')
            _, base_meta = verify_artifact(cfg, model, 'fp32')
            packed_meta = data['artifact_metadata']
            if (packed_meta['fp32_checkpoint_sha256'] != base_meta['fp32_checkpoint_sha256'] or
                    packed_meta['split_fingerprint'] != base_meta['split_fingerprint']):
                raise ValueError('Low-bit and baseline do not share checkpoint/splits')
            rows.append({'model': model, 'method': f'w{width}a32', 'accuracy_pct': data['accuracy'],
                         'macro_f1': data['macro_f1'], 'loss': data['loss'],
                         'serialized_size_mb': data['serialized_size_mb'], 'format': 'Packed .pth',
                         'compute': 'FP32 after unpacking'})
    table = pd.DataFrame(rows)
    for model in cfg['models']:
        base = table[(table.model == model) & (table.method == 'fp32')].iloc[0]
        mask = table.model == model
        table.loc[mask, 'delta_accuracy_pp'] = table.loc[mask, 'accuracy_pct'] - base.accuracy_pct
        table.loc[mask, 'delta_macro_f1'] = table.loc[mask, 'macro_f1'] - base.macro_f1
    table.to_csv(ROOT / 'results/classification_storage_comparison.csv', index=False)
    figures = ROOT / 'figures'
    figures.mkdir(exist_ok=True)
    for metric, ylabel in [('accuracy_pct', 'Test accuracy (%)'), ('macro_f1', 'Test Macro-F1'),
                           ('serialized_size_mb', 'Actual file size (MB; .pte or packed .pth)')]:
        methods = list(table.method.unique())
        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(len(cfg['models']))
        width = .8 / len(methods)
        for index, method in enumerate(methods):
            values = []
            for model in cfg['models']:
                matched = table[(table.model == model) & (table.method == method)]
                values.append(matched[metric].iloc[0] if len(matched) else np.nan)
            ax.bar(x + (index - (len(methods) - 1) / 2) * width, values, width, label=method)
        ax.set_xticks(x, cfg['models'])
        ax.legend()
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(figures / f'classification_storage_{metric}.png', dpi=160)
        plt.close(fig)
    text = ['# RealWaste: classification and storage', '', markdown_table(table), '',
            'All scores use the same frozen test split. W4A32/W2A32 quantize Conv/Linear weights only,',
            'store packed codes and scales, then unpack once to FP32. Activations and execution are FP32.',
            'File formats differ: packed .pth versus executable .pte. Size is actual serialized storage;',
            'it does not establish RAM reduction, native INT4 acceleration, or phone deployment.', '',
            '## Export drift on validation', '', markdown_table(pd.DataFrame(drift)), '',
            'runtime_validation permits measurement of executable INT8 models despite drift. It does not',
            'certify numerical equivalence or acceptable application quality. Full per-image logits remain',
            'in results/export_validation. This policy was revised after observing earlier validation failures;',
            'record that change when reporting the experiment. Test results are not used to tune this policy.', '',
            '## Evidence', '', 'Training curves: results/training/. Commands, versions, hashes: results/evidence/.',
            'Low-bit per-class metrics, predictions, confusion matrices: results/lowbit/.',
            'Mac M3/phone performance requires measurements on those devices; this report has no device latency claims.', '']
    (ROOT / 'docs').mkdir(exist_ok=True)
    (ROOT / 'docs/CLASSIFICATION_REPORT.md').write_text('\n'.join(text))
    print('Created docs/CLASSIFICATION_REPORT.md and results/classification_storage_comparison.csv', flush=True)
