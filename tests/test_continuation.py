"""Resume orchestration and real packed-weight evaluation on a small CNN."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import torch
from src.common import write_json


class LowbitExperimentTests(unittest.TestCase):
    def test_saved_packed_model_generates_metrics_predictions_and_can_rerun(self):
        from src import ptq_lowbit
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 2, 1), torch.nn.Flatten(), torch.nn.Linear(8, 2)).eval()
        export = types.ModuleType('src.export_edge')
        export.load_fp32 = lambda cfg, name: (copy.deepcopy(model), {'fp32_checkpoint_sha256': 'source', 'split_fingerprint': 'split'})
        data = types.ModuleType('src.data')
        batch = (torch.randn(2, 3, 2, 2), torch.tensor([0, 1]), ['a', 'b'])
        data.loader = lambda *args, **kw: [batch]
        data.loss_function = lambda *args: torch.nn.CrossEntropyLoss()
        cfg = {'seed': 42, 'classes': ['a', 'b'], 'preprocessing': {}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(ptq_lowbit, 'ROOT', Path(tmp)), \
                patch.object(ptq_lowbit, 'environment', return_value={'test': True}), \
                patch.dict(sys.modules, {'src.export_edge': export, 'src.data': data}):
            for bits in (4, 2):
                ptq_lowbit.experiment(cfg, 'mobilenet_v2', bits)
                out = Path(tmp) / f'results/lowbit/mobilenet_v2/w{bits}a32'
                self.assertTrue((out / 'test_confusion.png').exists())
                self.assertTrue((out / 'test_predictions.csv').exists())
                metric = json.loads((out / 'test_metrics.json').read_text())
                self.assertFalse(metric['native_lowbit_runtime'])
                self.assertEqual(metric['images'], 2)
                path = Path(tmp) / f'exported/mobilenet_v2/w{bits}a32_packed.pth'
                original = path.read_bytes()
                ptq_lowbit.experiment(cfg, 'mobilenet_v2', bits)
                self.assertEqual(path.read_bytes(), original)
                metadata = path.with_suffix('.json')
                value = json.loads(metadata.read_text()); value['sha256'] = 'corrupt'
                write_json(metadata, value)
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    ptq_lowbit.experiment(cfg, 'mobilenet_v2', bits)


class ContinuationTests(unittest.TestCase):
    def setup_run(self, root):
        from src import continue_experiments as c
        cfg = {'models': ['mobilenet_v2', 'mobilenet_v3_large', 'shufflenet_v2_x1_0']}
        for name in cfg['models']:
            folder = root / 'checkpoints' / name
            folder.mkdir(parents=True)
            (folder / 'training_summary.json').write_text('{}')
            (folder / 'qat_last.pth').write_bytes(b'saved')
            if name != 'mobilenet_v2':
                (folder / 'qat_summary.json').write_text('{}')
        export = types.ModuleType('src.export_edge')
        export.load_fp32 = lambda cfg, name: (None, {})
        return c, cfg, export

    def test_resume_export_only_and_skip_completed_fp32(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); c, cfg, export = self.setup_run(root)
            completed = {(m, 'fp32') for m in cfg['models']}
            calls = []
            def stage(name, args):
                calls.append((name, args))
                if name in ('qat_int8', 'ptq_int8'):
                    model = args[args.index('--model') + 1]
                    completed.add((model, 'qat' if name == 'qat_int8' else 'ptq'))
                return 0
            with patch.object(c, 'ROOT', root), patch.object(c, 'config', return_value=cfg), \
                    patch.object(c, 'artifact_done', side_effect=lambda cfg,m,s:(m,s) in completed), \
                    patch.object(c, 'run', side_effect=stage), patch.dict(sys.modules, {'src.export_edge': export}):
                c.continue_run('config.yaml')
            qat = [args for name, args in calls if name == 'qat_int8']
            self.assertIn('--resume', qat[0])
            self.assertTrue(all('--export-only' in args for args in qat[1:]))
            self.assertNotIn('train_fp32', [name for name, _ in calls])
            self.assertEqual(sum(name == 'ptq_lowbit' for name, _ in calls), 6)
            self.assertEqual(calls[-1][0], 'report')

    def test_failure_defers_test_and_records_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); c, cfg, export = self.setup_run(root)
            calls = []
            with patch.object(c, 'ROOT', root), patch.object(c, 'config', return_value=cfg), \
                    patch.object(c, 'artifact_done', return_value=False), \
                    patch.object(c, 'run', side_effect=lambda n,a: calls.append(n) or 1), \
                    patch.dict(sys.modules, {'src.export_edge': export}):
                with self.assertRaisesRegex(RuntimeError, 'deferred'):
                    c.continue_run('config.yaml')
            self.assertNotIn('evaluate', calls)
            self.assertNotIn('ptq_lowbit', calls)
            status = json.loads(next((root / 'results/continuations').glob('*.json')).read_text())
            self.assertEqual(status['status'], 'failed')
            self.assertTrue(status['failures'])


class ClassificationReportTests(unittest.TestCase):
    def test_report_works_without_device_benchmark_and_includes_lowbit(self):
        import pandas as pd
        evaluate = types.ModuleType('src.evaluate')
        evaluate.verify_artifact = lambda cfg, name, method: (None, {
            'pte_sha256': 'baseline-artifact', 'fp32_checkpoint_sha256': 'source',
            'split_fingerprint': 'split', 'parity': {'policy': 'runtime_validation',
            'validation_argmax_agreement': .99, 'runtime_minus_reference_accuracy_pp': -.2}})
        module_patch = patch.dict(sys.modules, {'src.evaluate': evaluate})
        module_patch.start()
        self.addCleanup(module_patch.stop)
        sys.modules.pop('src.quality_report', None)
        report = importlib.import_module('src.quality_report')
        cfg = {'models': ['mobilenet_v2']}
        rows = [{'model': 'mobilenet_v2', 'method': m, 'accuracy_pct': 90., 'macro_f1': .9,
                 'loss': .2, 'serialized_size_mb': 8. if m == 'fp32' else 2.,
                 'pte_sha256': 'baseline-artifact'} for m in ('fp32', 'ptq', 'qat')]
        with tempfile.TemporaryDirectory() as tmp, patch.object(report, 'ROOT', Path(tmp)):
            root = Path(tmp)
            (root / 'results').mkdir()
            path = root / 'exported/mobilenet_v2/w4a32_packed.pth'
            path.parent.mkdir(parents=True); path.write_bytes(b'packed')
            from src.common import sha256
            write_json(root / 'results/lowbit/mobilenet_v2/w4a32/test_metrics.json', {
                'accuracy': 80., 'macro_f1': .8, 'loss': .4, 'serialized_size_mb': 1.,
                'packed_sha256': sha256(path), 'artifact_metadata': {
                    'fp32_checkpoint_sha256': 'source', 'split_fingerprint': 'split'}})
            report.quality_report(cfg, pd.DataFrame(rows))
            table = pd.read_csv(root / 'results/classification_storage_comparison.csv')
            self.assertEqual(len(table), 4)
            self.assertEqual(table[table.method == 'w4a32'].iloc[0].delta_accuracy_pp, -10.)
            self.assertTrue((root / 'docs/CLASSIFICATION_REPORT.md').exists())
            self.assertTrue((root / 'figures/classification_storage_accuracy_pct.png').exists())
        sys.modules.pop('src.quality_report', None)


if __name__ == '__main__':
    unittest.main()
