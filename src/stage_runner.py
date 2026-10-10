"""Run a project stage with durable console logs and provenance, including failures."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('verify_data', 'train_fp32', 'evaluate_checkpoint', 'export_edge', 'ptq_int8',
          'qat_int8', 'ptq_lowbit', 'continue_experiments', 'evaluate', 'training_history', 'benchmark', 'report', 'smoke_test', 'run_kaggle', 'run_pipeline')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def option(args, name, default=None):
    for i, item in enumerate(args):
        if item == name and i + 1 < len(args):
            return args[i + 1]
        if item.startswith(name + '='):
            return item.split('=', 1)[1]
    return default


def run(stage, args):
    if stage not in STAGES:
        raise ValueError(f'Unknown stage: {stage}')
    model = option(args, '--model', 'all')
    if model not in ('all', 'mobilenet_v2', 'mobilenet_v3_large', 'shufflenet_v2_x1_0'):
        raise ValueError(f'Unknown model: {model}')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out = ROOT / 'results/evidence' / model / stage / stamp
    out.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-u', '-m', 'src.' + stage, *args]
    metadata = {'stage': stage, 'model': model, 'command': command,
                'started_at_utc': datetime.now(timezone.utc).isoformat(),
                'status': 'running', 'cwd': str(ROOT), 'python': sys.version,
                'platform': platform.platform(), 'versions': {}}
    for package in ('torch', 'torchvision', 'torchao', 'executorch', 'numpy', 'Pillow',
                    'scikit-learn', 'matplotlib', 'PyYAML', 'psutil'):
        try:
            metadata['versions'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            metadata['versions'][package] = None
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    metadata['git_commit'] = revision.stdout.strip() if revision.returncode == 0 else None
    dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, capture_output=True, text=True)
    metadata['git_dirty'] = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    config_path = Path(option(args, '--config', 'configs/kaggle.yaml' if stage == 'run_kaggle' else 'configs/project.yaml'))
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if config_path.is_file():
        (out / 'config.yaml').write_bytes(config_path.read_bytes())
        metadata['config_sha256'] = digest(config_path)
        metadata['config_path'] = str(config_path)
    inputs = [*sorted((ROOT / 'src').glob('*.py')), *sorted((ROOT / 'data/splits').glob('*')),
              ROOT / 'data/data_audit.csv', *sorted(ROOT.glob('requirements*.txt'))]
    metadata['input_sha256'] = {str(p.relative_to(ROOT)): digest(p) for p in inputs if p.is_file()}
    if model != 'all':
        for path in (ROOT / 'checkpoints' / model).glob('*.pth'):
            metadata['input_sha256'][str(path.relative_to(ROOT))] = digest(path)
    write(out / 'stage.json', metadata)
    print(f'Evidence: {out}', flush=True)
    started = time.perf_counter()
    wall_start = time.time_ns()
    returncode = 1
    process = None
    try:
        # Child stdout/stderr are captured at OS level, including native runtime messages.
        with (out / 'console.log').open('w', buffering=1) as log:
            child_env = dict(os.environ, WASTE_EDGE_RECORDED_STAGE=stage)
            process = subprocess.Popen(command, cwd=ROOT, env=child_env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, errors='replace')
            with process.stdout:
                for line in process.stdout:
                    log.write(line)
                    print(line, end='', flush=True)
            returncode = process.wait()
        metadata['status'] = 'completed' if returncode == 0 else 'failed'
    except BaseException as exc:
        metadata['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        metadata['error'] = repr(exc)
        with (out / 'console.log').open('a') as log:
            log.write(f'\nStage runner error: {exc!r}\n')
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        metadata.update(exit_code=process.returncode if process else returncode,
                        seconds=time.perf_counter() - started,
                        finished_at_utc=datetime.now(timezone.utc).isoformat())
        # Keep identities of outputs produced in this invocation, not unrelated old files.
        outputs = {}
        for folder in ('checkpoints', 'exported', 'results', 'figures', 'docs'):
            for path in (ROOT / folder).rglob('*'):
                if (path.is_file() and 'evidence' not in path.relative_to(ROOT).parts
                        and path.stat().st_mtime_ns >= wall_start):
                    outputs[str(path.relative_to(ROOT))] = {'sha256': digest(path), 'bytes': path.stat().st_size}
        metadata['outputs'] = outputs
        write(out / 'stage.json', metadata)
    if returncode == 0 and stage == 'run_kaggle':
        # Package only after the pipeline's own evidence record has been finalized.
        from .package_results import package_results
        package_results()
    return returncode


def logged_main(stage, main_function):
    """Make existing direct CLI commands record evidence too, without recursive launches."""
    if os.environ.get('WASTE_EDGE_RECORDED_STAGE') == stage:
        return main_function()
    raise SystemExit(run(stage, sys.argv[1:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    raise SystemExit(run(args.stage, args.args))


if __name__ == '__main__':
    main()
