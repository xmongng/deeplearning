"""Continue saved FP32/QAT experiments sequentially without retraining FP32."""
import argparse
from datetime import datetime, timezone
from .common import ROOT, config, write_json
from .stage_runner import run


def artifact_done(cfg, model, method):
    path = ROOT / 'exported' / model / f'{method}.pte'
    meta = path.with_suffix('.json')
    if not path.exists() or not meta.exists():
        return False
    from .evaluate import verify_artifact
    verify_artifact(cfg, model, method)
    return True


def continue_run(config_path, device='cuda', bits=(4, 2)):
    cfg = config(config_path)
    failures = []
    def stage(name, *args):
        code = run(name, ['--config', config_path, *args])
        if code:
            failures.append({'stage': name, 'args': args, 'exit_code': code})
        return code == 0
    status = ROOT / 'results/continuations' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '.json')
    error = None
    try:
        # Verify all baseline sources before any new stage or test evaluation.
        from .export_edge import load_fp32
        for model in cfg['models']:
            if not (ROOT / 'checkpoints' / model / 'training_summary.json').exists():
                raise ValueError(f'{model}: complete FP32 training first')
            load_fp32(cfg, model)
        for model in cfg['models']:
            for method, command in [('fp32', 'export_edge'), ('ptq', 'ptq_int8')]:
                if not artifact_done(cfg, model, method):
                    stage(command, '--model', model)
            if not artifact_done(cfg, model, 'qat'):
                folder = ROOT / 'checkpoints' / model
                if (folder / 'qat_summary.json').exists():
                    stage('qat_int8', '--model', model, '--export-only')
                elif (folder / 'qat_last.pth').exists():
                    stage('qat_int8', '--model', model, '--resume', '--device', device)
                elif (folder / 'qat_best.pth').exists():
                    raise ValueError(f'{model}: missing qat_last.pth; cannot resume safely')
                else:
                    stage('qat_int8', '--model', model, '--device', device)
        missing = [f'{m}/{s}' for m in cfg['models'] for s in ('fp32', 'ptq', 'qat')
                   if not artifact_done(cfg, m, s)]
        if missing:
            raise RuntimeError('Test evaluation deferred; missing artifacts: ' + ', '.join(missing))
        if stage('evaluate'):
            for model in cfg['models']:
                for width in bits:
                    stage('ptq_lowbit', '--model', model, '--bits', str(width), '--device', device)
            stage('report', '--quality-only')
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        write_json(status, {'config': config_path, 'bits': list(bits), 'failures': failures,
                            'status': 'failed' if failures or error else 'completed', 'error': error,
                            'notes': 'INT8 runtime validation permits drift; packed low-bit inference uses FP32 kernels.'})
    if failures:
        raise RuntimeError(f'{len(failures)} stage(s) failed; see {status}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/kaggle.yaml')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--bits', nargs='+', type=int, choices=[2, 4], default=[4, 2])
    args = p.parse_args()
    continue_run(args.config, args.device, args.bits)


if __name__ == '__main__':
    from .stage_runner import logged_main
    logged_main('continue_experiments', main)
