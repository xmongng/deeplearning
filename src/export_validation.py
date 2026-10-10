"""Compare export outputs without discarding diagnostics at the first mismatch."""
import math

import torch
from sklearn.metrics import accuracy_score, f1_score


def _values(tensor):
    return [v if math.isfinite(v) else None for v in tensor.detach().cpu().reshape(-1).tolist()]


def compare_outputs(expected, observed, sample, classes, method, target=None):
    expected = expected.detach().cpu()
    observed = observed.detach().cpu()
    shape = (1, len(classes))
    valid = (tuple(expected.shape) == shape and tuple(observed.shape) == shape
             and expected.is_floating_point() and observed.is_floating_point()
             and bool(torch.isfinite(expected).all()) and bool(torch.isfinite(observed).all()))
    row = {'sample': sample, 'target': target, 'reference_shape': list(expected.shape),
           'runtime_shape': list(observed.shape), 'valid_output': valid,
           'reference_logits': _values(expected), 'runtime_logits': _values(observed),
           'atol': .001 if method == 'fp32' else .15, 'rtol': .001}
    if not valid:
        row.update(numerically_close=False, argmax_agreement=False)
        return row
    delta = (observed.double() - expected.double()).abs()
    close = torch.isclose(observed.double(), expected.double(), rtol=row['rtol'], atol=row['atol'])
    reference_prediction, runtime_prediction = expected.argmax(1).item(), observed.argmax(1).item()
    row.update(numerically_close=bool(close.all()), mismatched_elements=int((~close).sum()),
               max_abs_error=delta.max().item(), mean_abs_error=delta.mean().item(),
               rmse=delta.square().mean().sqrt().item(),
               max_relative_error=(delta / expected.double().abs().clamp_min(1e-12)).max().item(),
               reference_prediction=reference_prediction, runtime_prediction=runtime_prediction,
               argmax_agreement=reference_prediction == runtime_prediction)
    return row


def summarize_parity(rows, method, num_classes, smoke=False):
    if not rows:
        raise ValueError('No parity samples')
    validation = [r for r in rows if r['target'] is not None]
    valid = all(r['valid_output'] for r in rows)
    numerical_mismatches = sum(not r['numerically_close'] for r in rows)
    reasons = []
    if not valid:
        reasons.append('Invalid output shape or non-finite output')
    if method == 'fp32' or smoke:
        if numerical_mismatches:
            reasons.append('Strict numerical parity failed')
    elif not validation:
        reasons.append('INT8 export requires the complete validation set')
    elif any(not r['argmax_agreement'] for r in validation):
        reasons.append('INT8 runtime predictions differ from reference on validation')
    summary = {'status': 'failed' if reasons else 'passed', 'failure_reasons': reasons,
               'policy': 'strict_numerical' if method == 'fp32' or smoke else 'exact_validation_predictions',
               'numerical_mismatch_samples': numerical_mismatches, 'samples': len(rows),
               'validation_samples': len(validation),
               'max_abs_error': max((r['max_abs_error'] for r in rows if 'max_abs_error' in r), default=None),
               'argmax_agreement': sum(r['argmax_agreement'] for r in rows) / len(rows),
               'validation_argmax_agreement': (sum(r['argmax_agreement'] for r in validation) / len(validation)) if validation else None,
               'atol': .001 if method == 'fp32' else .15, 'rtol': .001,
               'notes': 'INT8 raw-logit closeness is diagnostic; all validation predictions must match exactly. Synthetic input is a structural probe, not a quality dataset. Test is untouched.'}
    if validation and all(r['valid_output'] for r in validation):
        targets = [r['target'] for r in validation]
        labels = list(range(num_classes))
        for key in ('reference', 'runtime'):
            predictions = [r[key + '_prediction'] for r in validation]
            summary[key + '_accuracy_pct'] = accuracy_score(targets, predictions) * 100
            summary[key + '_macro_f1'] = f1_score(targets, predictions, labels=labels, average='macro', zero_division=0)
        summary['runtime_minus_reference_accuracy_pp'] = summary['runtime_accuracy_pct'] - summary['reference_accuracy_pct']
        summary['runtime_minus_reference_macro_f1'] = summary['runtime_macro_f1'] - summary['reference_macro_f1']
    return summary
