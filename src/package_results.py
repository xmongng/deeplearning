"""Package source, frozen protocol, metrics, figures, artifacts and completed stage logs."""
import argparse
from datetime import datetime, timezone
import json
import csv
from pathlib import Path
import shutil
import tempfile

from .stage_runner import ROOT, digest, write


def package_results(output=None):
    output = Path(output or ('/kaggle/working/waste-edge-complete.zip'
                             if Path('/kaggle/working').is_dir() else ROOT / 'waste-edge-complete.zip'))
    if output.suffix != '.zip':
        raise ValueError('Output must have .zip extension')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'Archive already exists: {output}; choose a new --output filename')
    required = ('checkpoints', 'exported', 'results/quality_metrics.csv')
    for name in required:
        if not (ROOT / name).exists():
            raise FileNotFoundError(f'Complete training/export/evaluation first: missing {name}')
    with (ROOT / 'results/quality_metrics.csv').open(newline='') as stream:
        quality = list(csv.DictReader(stream))
    expected = {(name, method) for name in ('mobilenet_v2', 'mobilenet_v3_large', 'shufflenet_v2_x1_0')
                for method in ('fp32', 'ptq', 'qat')}
    if len(quality) != 9 or {(r['model'], r['method']) for r in quality} != expected:
        raise ValueError('Expected quality metrics for all nine deployed artifacts')
    for row in quality:
        path = ROOT / 'exported' / row['model'] / (row['method'] + '.pte')
        metadata = json.loads(path.with_suffix('.json').read_text())
        if metadata.get('smoke_only') or digest(path) != row['pte_sha256'] or metadata['pte_sha256'] != row['pte_sha256']:
            raise ValueError(f'Quality metrics/artifact mismatch: {path}')
    with tempfile.TemporaryDirectory(prefix='waste-edge-package-', dir=ROOT.parent) as temp:
        package = Path(temp) / 'package'
        package.mkdir()
        for name in ('checkpoints', 'exported', 'results', 'data', 'configs', 'figures', 'docs', 'src', 'tests'):
            source = ROOT / name
            if source.is_dir():
                shutil.copytree(source, package / name,
                                ignore=shutil.ignore_patterns('raw', '__pycache__', '*.pyc'))
        for source in [ROOT / 'README.md', *ROOT.glob('requirements*.txt'), *ROOT.glob('environment*.json')]:
            if source.is_file():
                shutil.copy2(source, package / source.name)
        manifest = {p.relative_to(package).as_posix(): {'sha256': digest(p), 'bytes': p.stat().st_size}
                    for p in sorted(package.rglob('*')) if p.is_file()}
        write(package / 'package_manifest.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'files': manifest, 'notes': 'Raw dataset excluded; file hashes preserve the evidence chain.'})
        archive = shutil.make_archive(str(Path(temp) / 'archive'), 'zip', package)
        Path(archive).replace(output)
    receipt = {'archive': str(output), 'sha256': digest(output), 'bytes': output.stat().st_size,
               'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'completed'}
    write_path = ROOT / 'results/packaging' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '.json')
    write_path.parent.mkdir(parents=True, exist_ok=True)
    write(write_path, receipt)
    print(f'Created {output}\nSHA-256: {receipt["sha256"]}', flush=True)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output')
    args = p.parse_args()
    package_results(args.output)


if __name__ == '__main__':
    main()
