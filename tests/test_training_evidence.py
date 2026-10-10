"""Focused tests runnable with unittest, without ExecuTorch or a dataset download."""
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

from src import stage_runner
from src.training_metrics import run_epoch
from src.training_history import save_history, plot_history


class EpochMetricsTests(unittest.TestCase):
    def test_weighted_loss_is_independent_of_batch_partition_and_keeps_predictions(self):
        model = torch.nn.Identity()
        model._num_classes = 2
        logits = torch.tensor([[2., 1.], [0., 3.], [2., 0.], [1., 2.]])
        targets = torch.tensor([0, 1, 1, 1])
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 4.]))
        rows = []
        batches = [(logits[:1], targets[:1], ['a']), (logits[1:], targets[1:], ['b', 'c', 'd'])]
        metrics = run_epoch(model, batches, criterion, 'cpu', prediction_rows=rows)
        self.assertAlmostEqual(metrics['loss'], criterion(logits, targets).item(), places=6)
        self.assertEqual(metrics['accuracy'], 75.0)
        self.assertEqual([r['path'] for r in rows], ['a', 'b', 'c', 'd'])
        self.assertAlmostEqual(sum(rows[0][f'probability_{i}'] for i in range(2)), 1.0, places=6)

    def test_empty_epoch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Empty epoch'):
            run_epoch(torch.nn.Identity(), [], torch.nn.CrossEntropyLoss(), 'cpu')

    def test_failed_serialization_keeps_previous_completed_evidence(self):
        from src.common import write_json, write_csv
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'history.json'
            write_json(path, {'completed_epoch': 1})
            with self.assertRaises(ValueError):
                write_json(path, {'loss': float('nan')})
            self.assertEqual(json.loads(path.read_text()), {'completed_epoch': 1})
            csv_path = Path(temp) / 'history.csv'
            write_csv(csv_path, [{'epoch': 1}])
            original = csv_path.read_text()
            with self.assertRaises(ValueError):
                write_csv(csv_path, [{'unexpected': 2}], fields=['epoch'])
            self.assertEqual(csv_path.read_text(), original)

    def test_history_can_regenerate_both_figure_formats(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'mobilenet_v2/fp32'
            rows = [{'epoch': 1, 'learning_rate': .0003, 'seconds': 1,
                     'train_loss': 1.3, 'train_accuracy': 50, 'train_macro_f1': .4,
                     'val_loss': 1.4, 'val_accuracy': 45, 'val_macro_f1': .35}]
            save_history(out, rows)
            plot_history(out)
            self.assertEqual(json.loads((out / 'epochs.json').read_text()), rows)
            self.assertGreater((out / 'training_curves.png').stat().st_size, 1000)
            self.assertTrue((out / 'training_curves.pdf').is_file())


class StageEvidenceTests(unittest.TestCase):
    def test_success_and_failure_capture_native_streams_and_distinct_invocations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'src').mkdir()
            (root / 'src/__init__.py').write_text('')
            (root / 'src/verify_data.py').write_text(
                'import sys\nfrom pathlib import Path\n'
                'print("stdout evidence", flush=True)\n'
                'print("stderr evidence", file=sys.stderr, flush=True)\n'
                'Path("results/example.txt").write_text("output")\n'
                'if "--fail" in sys.argv: raise RuntimeError("expected failure")\n')
            (root / 'configs').mkdir()
            (root / 'configs/project.yaml').write_text('seed: 42\n')
            with patch.object(stage_runner, 'ROOT', root):
                self.assertEqual(stage_runner.run('verify_data', []), 0)
                self.assertNotEqual(stage_runner.run('verify_data', ['--fail']), 0)
            runs = sorted((root / 'results/evidence/all/verify_data').iterdir())
            self.assertEqual(len(runs), 2)
            first, failed = [json.loads((p / 'stage.json').read_text()) for p in runs]
            self.assertEqual(first['status'], 'completed')
            self.assertEqual(failed['status'], 'failed')
            self.assertIn('results/example.txt', first['outputs'])
            self.assertIn('src/verify_data.py', first['input_sha256'])
            self.assertEqual((runs[0] / 'config.yaml').read_text(), 'seed: 42\n')
            log = (runs[1] / 'console.log').read_text()
            self.assertIn('stdout evidence', log)
            self.assertIn('stderr evidence', log)
            self.assertIn('expected failure', log)


class TrainingPersistenceTests(unittest.TestCase):
    def test_fp32_training_persists_metrics_config_checkpoints_and_figures(self):
        data = types.ModuleType('src.data')
        models = types.ModuleType('src.models')
        batch = (torch.randn(4, 3), torch.tensor([0, 1, 0, 1]), ['a', 'b', 'c', 'd'])
        data.loader = lambda *args, **kwargs: [batch]
        data.loss_function = lambda cfg, device: torch.nn.CrossEntropyLoss()
        models.NAMES = ('mobilenet_v2',)
        models.build_model = lambda name, num_classes, pretrained: torch.nn.Linear(3, num_classes)
        with patch.dict(sys.modules, {'src.data': data, 'src.models': models}):
            sys.modules.pop('src.train_fp32', None)
            training = importlib.import_module('src.train_fp32')
        cfg = {'seed': 42, 'classes': ['a', 'b'], 'preprocessing': {},
               'train': {'device': 'cpu', 'batch_size': 4, 'learning_rate': .001,
                         'weight_decay': .0001, 'epochs': 2, 'patience': 5}}
        with tempfile.TemporaryDirectory() as temp, patch.object(training, 'ROOT', Path(temp)), \
                patch.object(training, 'environment', return_value={'test': True}), \
                patch.object(training, 'split_fingerprint', return_value='frozen'):
            training.train(cfg, 'mobilenet_v2', smoke=True)
            out = Path(temp) / 'results/smoke/training/mobilenet_v2/fp32'
            rows = json.loads((out / 'epochs.json').read_text())
            self.assertEqual(len(rows), 2)
            self.assertIn('train_loss', rows[0])
            self.assertIn('val_loss', rows[0])
            self.assertTrue((out / 'training_curves.png').is_file())
            summary = json.loads((out / 'summary.json').read_text())
            self.assertEqual(summary['completed_epochs'], 2)
            self.assertEqual(summary['split_fingerprint'], 'frozen')
            checkpoint = Path(temp) / 'results/smoke/checkpoints/mobilenet_v2/fp32_last.pth'
            state = torch.load(checkpoint, weights_only=True)
            self.assertIn('optimizer', state)
            self.assertIn('scheduler', state)
            self.assertIn('rng_state', state)
        sys.modules.pop('src.train_fp32', None)


class QatPersistenceTests(unittest.TestCase):
    def test_qat_history_records_lr_freeze_flags_and_last_optimizer_state(self):
        data = types.ModuleType('src.data')
        models = types.ModuleType('src.models')
        export = types.ModuleType('src.export_edge')
        quantization = types.ModuleType('src.quantization')
        pt2e = types.ModuleType('torchao.quantization.pt2e')
        batch = (torch.randn(4, 3), torch.tensor([0, 1, 0, 1]), ['a', 'b', 'c', 'd'])
        data.loader = lambda *args, **kwargs: [batch]
        data.loss_function = lambda cfg, device: torch.nn.CrossEntropyLoss()
        models.NAMES = ('mobilenet_v2',)
        export.load_fp32 = lambda cfg, name: (torch.nn.Linear(3, 2),
              {'fp32_checkpoint_sha256': 'baseline', 'split_fingerprint': 'frozen'})
        exported = []
        export.export_model = lambda *args, **kwargs: exported.append(args)
        quantization.prepare = lambda model, cfg, qat, device: model.to(device).train(qat)
        quantization.convert = lambda model: copy.deepcopy(model).eval()
        quantization.freeze_bn = lambda model: None
        pt2e.disable_observer = lambda model: None
        # Stand-in quantization backend verifies persistence/control flow, not INT8 numerics.
        with patch.dict(sys.modules, {'src.data': data, 'src.models': models,
                'src.export_edge': export, 'src.quantization': quantization,
                'torchao.quantization.pt2e': pt2e}):
            sys.modules.pop('src.qat_int8', None)
            qat = importlib.import_module('src.qat_int8')
        cfg = {'seed': 42, 'classes': ['a', 'b'], 'preprocessing': {},
               'train': {'weight_decay': .0001},
               'qat': {'device': 'cpu', 'batch_size': 4, 'learning_rate': .001,
                       'epochs': 2, 'freeze_observer_epoch': 1, 'freeze_bn_epoch': 0}}
        with tempfile.TemporaryDirectory() as temp, patch.object(qat, 'ROOT', Path(temp)), \
                patch.object(qat, 'environment', return_value={'test': True}), \
                patch.object(qat, 'split_fingerprint', return_value='frozen'):
            (Path(temp) / 'checkpoints/mobilenet_v2').mkdir(parents=True)
            original_run_epoch = qat.run_epoch
            calls = [0]
            def interrupted_epoch(*args, **kwargs):
                calls[0] += 1
                if calls[0] == 3:
                    raise KeyboardInterrupt()
                return original_run_epoch(*args, **kwargs)
            with patch.object(qat, 'run_epoch', side_effect=interrupted_epoch):
                with self.assertRaises(KeyboardInterrupt):
                    qat.train_qat(cfg, 'mobilenet_v2')
            partial = torch.load(Path(temp) / 'checkpoints/mobilenet_v2/qat_last.pth', weights_only=True)
            self.assertEqual(partial['epoch'], 1)
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                changed = copy.deepcopy(cfg)
                changed['qat']['batch_size'] = 2
                qat.train_qat(changed, 'mobilenet_v2', resume=True)
            qat.train_qat(cfg, 'mobilenet_v2', resume=True)
            out = Path(temp) / 'results/training/mobilenet_v2/qat'
            self.assertEqual([r['epoch'] for r in json.loads((out / 'epochs.json').read_text())], [1, 2])
            self.assertTrue((out / 'resume_from_epoch_1.json').exists())
            rows = json.loads((out / 'epochs.json').read_text())
            self.assertEqual([r['observer_frozen'] for r in rows], [False, True])
            self.assertTrue(all(r['batchnorm_frozen'] for r in rows))
            self.assertIn('val_int8_loss', rows[0])
            self.assertEqual(rows[0]['learning_rate'], .001)
            self.assertTrue((out / 'training_curves.pdf').is_file())
            state = torch.load(Path(temp) / 'checkpoints/mobilenet_v2/qat_last.pth', weights_only=True)
            self.assertIn('optimizer', state)
            self.assertIn('rng_state', state)
            self.assertEqual(len(exported), 1)
        sys.modules.pop('src.qat_int8', None)


class PtqEvidenceTests(unittest.TestCase):
    def test_calibration_saves_paths_observer_state_and_export_summary(self):
        data = types.ModuleType('src.data')
        models = types.ModuleType('src.models')
        models.NAMES = ('mobilenet_v2',)
        export = types.ModuleType('src.export_edge')
        quantization = types.ModuleType('src.quantization')
        model = torch.nn.Linear(3, 2)
        model.register_buffer('observer_min', torch.tensor([0.]))
        data.loader = lambda cfg, split: [(torch.ones(1, 3), torch.tensor([1]), ['image.jpg'])]
        export.load_fp32 = lambda cfg, name: (model, {'fp32_checkpoint_sha256': 'baseline'})
        export.export_model = lambda *args: {'model': 'mobilenet_v2', 'method': 'ptq', 'pte_sha256': 'artifact'}
        quantization.prepare = lambda model, cfg: model
        quantization.convert = lambda model: model
        with patch.dict(sys.modules, {'src.data': data, 'src.models': models, 'src.export_edge': export, 'src.quantization': quantization}):
            sys.modules.pop('src.ptq_int8', None)
            ptq = importlib.import_module('src.ptq_int8')
        with tempfile.TemporaryDirectory() as temp, patch.object(ptq, 'ROOT', Path(temp)), \
                patch.object(ptq, 'config', return_value={'seed': 42}), \
                patch.object(ptq, 'environment', return_value={'test': True}), \
                patch.object(ptq, 'read_csv', return_value=[{'path': 'image.jpg'}]), \
                patch.object(sys, 'argv', ['ptq', '--model', 'mobilenet_v2']):
            ptq.main()
            out = Path(temp) / 'results/quantization/mobilenet_v2/ptq'
            self.assertIn('image.jpg', (out / 'calibration.csv').read_text())
            state = torch.load(out / 'observer_state.pth', weights_only=True)
            self.assertIn('observer_min', state)
            self.assertEqual(json.loads((out / 'summary.json').read_text())['calibration_images'], 1)
        sys.modules.pop('src.ptq_int8', None)


class BaselineEvaluationTests(unittest.TestCase):
    def test_best_checkpoint_evaluation_saves_predictions_and_never_reads_test(self):
        from src import common
        data = types.ModuleType('src.data')
        models = types.ModuleType('src.models')
        visited = []
        batch = (torch.tensor([[2., 0.], [0., 2.]]), torch.tensor([0, 1]), ['a', 'b'])
        def batches(cfg, split, batch_size, train):
            visited.append((split, train))
            return [batch]
        data.loader = batches
        data.loss_function = lambda cfg, device: torch.nn.CrossEntropyLoss()
        models.NAMES = ('mobilenet_v2',)
        models.build_model = lambda *args, **kwargs: torch.nn.Linear(2, 2)
        with patch.dict(sys.modules, {'src.data': data, 'src.models': models}):
            sys.modules.pop('src.evaluate_checkpoint', None)
            evaluation = importlib.import_module('src.evaluate_checkpoint')
        cfg = {'classes': ['a', 'b'], 'preprocessing': {}, 'train': {'batch_size': 2}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / 'best.pth'
            model = torch.nn.Linear(2, 2)
            with torch.no_grad():
                model.weight.copy_(torch.eye(2))
                model.bias.zero_()
            torch.save({'state_dict': model.state_dict(), 'classes': cfg['classes'],
                        'preprocessing': {}, 'split_fingerprint': 'frozen', 'epoch': 2}, checkpoint)
            with patch.object(evaluation, 'ROOT', root), \
                    patch.object(evaluation, 'checkpoint_path', return_value=checkpoint), \
                    patch.object(common, 'split_fingerprint', return_value='frozen'):
                summary = evaluation.evaluate_checkpoint(cfg, 'mobilenet_v2')
            self.assertEqual(visited, [('train', False), ('val', False)])
            self.assertEqual(summary['metrics']['val']['accuracy'], 100.0)
            out = root / 'results/baseline_validation/mobilenet_v2'
            self.assertIn('probability_0', (out / 'val_predictions.csv').read_text())
            self.assertTrue((out / 'val_confusion.png').is_file())
            self.assertTrue((out / 'train_per_class.json').is_file())
        sys.modules.pop('src.evaluate_checkpoint', None)


class PackagingTests(unittest.TestCase):
    def test_archive_contains_source_evidence_and_verified_file_manifest(self):
        import csv
        import zipfile
        from src import package_results
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'checkpoints').mkdir()
            (root / 'src').mkdir()
            (root / 'src/example.py').write_text('example = True\n')
            (root / 'results/evidence/all/evaluate/run').mkdir(parents=True)
            (root / 'results/evidence/all/evaluate/run/stage.json').write_text('{"status":"completed"}')
            (root / 'data/raw').mkdir(parents=True)
            (root / 'data/raw/excluded.jpg').write_bytes(b'raw')
            rows = []
            for model in ('mobilenet_v2', 'mobilenet_v3_large', 'shufflenet_v2_x1_0'):
                folder = root / 'exported' / model
                folder.mkdir(parents=True)
                for method in ('fp32', 'ptq', 'qat'):
                    path = folder / (method + '.pte')
                    path.write_bytes(b'artifact')
                    checksum = stage_runner.digest(path)
                    path.with_suffix('.json').write_text(json.dumps({'pte_sha256': checksum, 'smoke_only': False}))
                    rows.append({'model': model, 'method': method, 'pte_sha256': checksum})
            with (root / 'results/quality_metrics.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            output = root / 'complete.zip'
            with patch.object(package_results, 'ROOT', root):
                receipt = package_results.package_results(output)
                with self.assertRaises(FileExistsError):
                    package_results.package_results(output)
                (root / 'exported/mobilenet_v2/fp32.pte').write_bytes(b'modified')
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    package_results.package_results(root / 'tampered.zip')
                self.assertFalse((root / 'tampered.zip').exists())
            self.assertEqual(receipt['sha256'], stage_runner.digest(output))
            with zipfile.ZipFile(output) as archive:
                self.assertIn('src/example.py', archive.namelist())
                self.assertIn('results/evidence/all/evaluate/run/stage.json', archive.namelist())
                self.assertNotIn('data/raw/excluded.jpg', archive.namelist())
                manifest = json.loads(archive.read('package_manifest.json'))
                import hashlib
                for name, information in manifest['files'].items():
                    self.assertEqual(information['sha256'], hashlib.sha256(archive.read(name)).hexdigest())


if __name__ == '__main__':
    unittest.main()
