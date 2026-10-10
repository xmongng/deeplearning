"""Export regression tests with a stand-in lowering backend; no ExecuTorch install needed."""
import importlib
import json
from pathlib import Path
import tempfile
import types
import sys
import unittest
from unittest.mock import patch

import torch

from src.export_validation import compare_outputs, summarize_parity


class ParityPolicyTests(unittest.TestCase):
    def sample(self, expected, observed, method='ptq', target=0):
        return compare_outputs(torch.tensor([expected], dtype=torch.float32),
                               torch.tensor([observed], dtype=torch.float32), 'example.jpg',
                               ['a', 'b', 'c'], method, target)

    def test_original_point_16_error_is_diagnostic_when_predictions_are_identical(self):
        row = self.sample([4., 2., 0.], [4., 2.159983, 0.])
        self.assertFalse(row['numerically_close'])
        self.assertGreater(row['max_abs_error'], .15)
        summary = summarize_parity([row], 'ptq', 3)
        self.assertEqual(summary['status'], 'passed')
        self.assertEqual(summary['numerical_mismatch_samples'], 1)
        self.assertEqual(summary['runtime_minus_reference_accuracy_pp'], 0)

    def test_int8_prediction_change_fails_even_when_logits_are_numerically_close(self):
        row = self.sample([1., .99, 0.], [.99, 1., 0.])
        self.assertTrue(row['numerically_close'])
        self.assertEqual(summarize_parity([row], 'ptq', 3)['status'], 'failed')

    def test_runtime_validation_records_drift_without_claiming_equivalence(self):
        row = self.sample([1., .99, 0.], [.99, 1., 0.])
        report = summarize_parity([row], 'ptq', 3, policy='runtime_validation')
        self.assertEqual(report['status'], 'passed')
        self.assertFalse(report['equivalent_predictions'])
        self.assertEqual(report['runtime_minus_reference_accuracy_pp'], -100)
        self.assertTrue(report['warnings'])
        invalid = self.sample([1., .99, 0.], [float('nan'), 1., 0.])
        self.assertEqual(summarize_parity([invalid], 'qat', 3, policy='runtime_validation')['status'], 'failed')

    def test_fp32_and_smoke_keep_strict_numerical_checks(self):
        for method, smoke in [('fp32', False), ('ptq', True)]:
            row = self.sample([4., 2., 0.], [4., 2.16, 0.], method)
            self.assertEqual(summarize_parity([row], method, 3, smoke)['status'], 'failed')

    def test_nonfinite_and_wrong_shape_are_serializable_and_fail(self):
        row = self.sample([4., 2., 0.], [float('nan'), 2., 0.])
        json.dumps(row, allow_nan=False)
        self.assertEqual(summarize_parity([row], 'qat', 3)['status'], 'failed')
        row = self.sample([4., 2., 0.], [4., 2.])
        self.assertFalse(row['valid_output'])

    def test_int8_cannot_pass_using_synthetic_probe_alone(self):
        row = self.sample([4., 2., 0.], [4., 2., 0.], target=None)
        self.assertEqual(summarize_parity([row], 'ptq', 3)['status'], 'failed')


class ExportPublicationTests(unittest.TestCase):
    def setUp(self):
        names = [
            'executorch.backends.xnnpack.partition.xnnpack_partitioner',
            'executorch.backends.xnnpack.utils.configs', 'executorch.exir',
            'executorch.runtime', 'src.models', 'src.quantization', 'src.data']
        modules = {name: types.ModuleType(name) for name in names}
        modules[names[0]].XnnpackPartitioner = lambda: object()
        modules[names[1]].get_transform_passes = lambda: []
        modules['executorch.exir'].to_edge_transform_and_lower = lambda *args, **kwargs: None
        modules['executorch.runtime'].Runtime = object
        modules['src.models'].NAMES = ('mobilenet_v2',)
        modules['src.models'].build_model = lambda *args, **kwargs: None
        modules['src.quantization'].example = lambda cfg: torch.tensor([[4., 2., 0.]])
        self.batches = [(torch.tensor([[4., 2., 0.]]), torch.tensor([0]), ['one.jpg']),
                        (torch.tensor([[0., 4., 1.]]), torch.tensor([1]), ['two.jpg'])]
        modules['src.data'].loader = lambda cfg, split: self.batches if split == 'val' else self.fail('Only validation is allowed')
        self.modules = modules
        with patch.dict(sys.modules, modules):
            sys.modules.pop('src.export_edge', None)
            self.export = importlib.import_module('src.export_edge')
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.out = self.root / 'exported/mobilenet_v2'
        self.cfg = {'classes': ['a', 'b', 'c'], 'preprocessing': {}}
        node = types.SimpleNamespace(op='call_function', target='quantized_decomposed.quantize_per_tensor.default')
        graph = types.SimpleNamespace(nodes=[node])
        ep = types.SimpleNamespace(graph=graph)
        delegate = types.SimpleNamespace(backend_id='XnnpackBackend',
                                          original_module=types.SimpleNamespace(graph=graph))
        edge = types.SimpleNamespace(graph=graph, modules=lambda: [delegate])
        lowered = types.SimpleNamespace(exported_program=lambda: types.SimpleNamespace(graph_module=edge),
                                        to_executorch=lambda: types.SimpleNamespace(buffer=b'candidate artifact'))
        self.patches = [patch.object(self.export, 'ROOT', self.root),
                       patch.object(self.export, 'artifact_dir', return_value=self.out),
                       patch.object(self.export, 'environment', return_value={'test': True}),
                       patch.object(self.export, 'to_edge_transform_and_lower', return_value=lowered),
                       patch.object(self.export.torch.export, 'export', return_value=ep),
                       patch.object(self.export.torch.export, 'save', side_effect=lambda ep, p: p.write_bytes(b'graph'))]
        for context in self.patches:
            context.start()

    def tearDown(self):
        for context in reversed(self.patches):
            context.stop()
        self.temp.cleanup()
        sys.modules.pop('src.export_edge', None)

    def run_export(self, change_prediction=False):
        def execute(inputs):
            result = inputs[0].clone()
            if change_prediction:
                result[:] = torch.tensor([[0., 0., 4.]])
            else:
                result[:, 2] += .16  # exceeds old logit tolerance, preserves these labels
            return (result,)
        runtime = types.SimpleNamespace(execute=execute)
        with patch.dict(sys.modules, self.modules), patch.object(self.export, 'load_runtime', return_value=(object(), runtime)):
            return self.export.export_model(torch.nn.Identity(), self.cfg, 'mobilenet_v2', 'ptq', {})

    def test_success_checks_all_validation_and_publishes_metadata_last(self):
        report = self.run_export()
        self.assertEqual(report['export_status'], 'passed')
        self.assertEqual(report['parity']['validation_samples'], len(self.batches))
        self.assertTrue((self.out / 'ptq.pte').is_file())
        self.assertEqual(json.loads((self.out / 'ptq.json').read_text())['parity_status'], 'passed')
        with self.assertRaisesRegex(RuntimeError, 'already exists'):
            self.run_export()

    def test_failure_keeps_candidate_and_report_and_allows_retry(self):
        with self.assertRaisesRegex(RuntimeError, 'parity failed'):
            self.run_export(change_prediction=True)
        self.assertFalse((self.out / 'ptq.pte').exists())
        self.assertFalse((self.out / 'ptq.json').exists())
        attempts = list((self.root / 'results/export_validation/mobilenet_v2/ptq').iterdir())
        failed = json.loads((attempts[0] / 'export_report.json').read_text())
        self.assertEqual(failed['export_status'], 'failed')
        self.assertEqual(len(json.loads((attempts[0] / 'parity_samples.json').read_text())), 3)
        self.assertTrue((attempts[0] / 'ptq.pte').is_file())
        self.run_export()
        self.assertTrue((self.out / 'ptq.pte').is_file())

    def test_lowering_failure_still_saves_report(self):
        with patch.object(self.export, 'to_edge_transform_and_lower', side_effect=RuntimeError('lowering failed')):
            with self.assertRaisesRegex(RuntimeError, 'lowering failed'):
                self.run_export()
        reports = list((self.root / 'results/export_validation').rglob('export_report.json'))
        report = json.loads(reports[0].read_text())
        self.assertEqual(report['export_status'], 'failed')
        self.assertEqual(report['error'], 'lowering failed')
        self.assertFalse((self.out / 'ptq.pte').exists())

    def test_publication_failure_leaves_no_final_artifact_pair(self):
        write_json = self.export.write_json
        def fail_final_metadata(path, value):
            if path == self.out / 'ptq.json':
                raise OSError('metadata write failed')
            return write_json(path, value)
        with patch.object(self.export, 'write_json', side_effect=fail_final_metadata):
            with self.assertRaisesRegex(OSError, 'metadata write failed'):
                self.run_export()
        self.assertFalse((self.out / 'ptq.pte').exists())
        self.assertFalse((self.out / 'ptq.json').exists())
        reports = list((self.root / 'results/export_validation').rglob('export_report.json'))
        self.assertEqual(json.loads(reports[0].read_text())['export_status'], 'failed')
        self.run_export()

    def test_partial_copy_failure_is_removed_from_final_directory(self):
        copy_file = self.export.shutil.copy2
        def fail_pte_copy(source, destination):
            if destination == self.out / 'ptq.pte':
                destination.write_bytes(b'partial')
                raise OSError('copy interrupted')
            return copy_file(source, destination)
        with patch.object(self.export.shutil, 'copy2', side_effect=fail_pte_copy):
            with self.assertRaisesRegex(OSError, 'copy interrupted'):
                self.run_export()
        self.assertFalse((self.out / 'ptq.pte').exists())
        self.assertFalse((self.out / 'ptq.pt2').exists())
        self.assertFalse((self.out / 'ptq.json').exists())
        self.assertTrue(list((self.root / 'results/export_validation').rglob('unpublished_ptq.pte')))

    def test_old_orphan_is_preserved_without_overwriting_a_completed_export(self):
        self.out.mkdir(parents=True)
        (self.out / 'ptq.pte').write_bytes(b'old incomplete artifact')
        self.run_export()
        recovered = list((self.root / 'results/export_validation').rglob('legacy_incomplete/ptq.pte'))
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0].read_bytes(), b'old incomplete artifact')
        self.assertEqual((self.out / 'ptq.pte').read_bytes(), b'candidate artifact')


if __name__ == '__main__':
    unittest.main()
