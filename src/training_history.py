"""Persist epoch metrics and regenerate training figures without retraining."""
import argparse

from .common import ROOT, read_csv, write_csv, write_json


def print_epoch(name, phase, row, count):
    val = 'val_int8' if phase == 'qat' else 'val'
    print(f"{name} {phase.upper()} epoch {row['epoch']}/{count} | "
          f"train loss={row['train_loss']:.5f} acc={row['train_accuracy']:.2f}% F1={row['train_macro_f1']:.4f} | "
          f"{val} loss={row[val + '_loss']:.5f} acc={row[val + '_accuracy']:.2f}% F1={row[val + '_macro_f1']:.4f} | "
          f"lr={row['learning_rate']:.3g} seconds={row['seconds']:.1f}", flush=True)


def save_history(out, rows):
    write_csv(out / 'epochs.csv', rows)
    write_json(out / 'epochs.json', rows)


def plot_history(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = read_csv(out / 'epochs.csv')
    if not rows:
        raise ValueError('Empty epoch history')
    val = 'val_int8' if 'val_int8_loss' in rows[0] else 'val'
    epochs = [int(r['epoch']) for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric, label in zip(axes.flat[:3], ('loss', 'accuracy', 'macro_f1'),
                                ('Cross-entropy loss', 'Accuracy (%)', 'Macro-F1')):
        ax.plot(epochs, [float(r['train_' + metric]) for r in rows], label='Train (augmentation, training mode)')
        ax.plot(epochs, [float(r[val + '_' + metric]) for r in rows], label='Validation INT8' if val == 'val_int8' else 'Validation')
        ax.set(xlabel='Epoch', ylabel=label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    axes.flat[3].plot(epochs, [float(r['learning_rate']) for r in rows])
    axes.flat[3].set(xlabel='Epoch', ylabel='Learning rate')
    axes.flat[3].grid(alpha=0.25)
    fig.suptitle(f'{out.parent.name} / {out.name}')
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'training_curves.{suffix}', dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--phase', choices=['fp32', 'qat'], required=True)
    args = p.parse_args()
    plot_history(ROOT / 'results/training' / args.model / args.phase)


if __name__ == '__main__':
    from .stage_runner import logged_main
    logged_main('training_history', main)
