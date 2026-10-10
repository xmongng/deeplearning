"""Epoch metrics with correctly aggregated weighted cross-entropy."""
import torch
from sklearn.metrics import accuracy_score, f1_score


def run_epoch(model, batches, criterion, device, optimizer=None, max_batches=None, prediction_rows=None):
    predictions, targets = [], []
    total_loss, total, loss_denominator = 0.0, 0, 0.0
    with torch.set_grad_enabled(optimizer is not None):
        for i, (x, y, paths) in enumerate(batches):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            # CrossEntropyLoss(weight=..., reduction="mean") divides by sum of target weights.
            denominator = criterion.weight[y].sum().item() if criterion.weight is not None else len(y)
            total_loss += loss.item() * denominator
            loss_denominator += denominator
            total += len(y)
            if prediction_rows is not None:
                detached = logits.detach().cpu()
                probabilities = detached.softmax(1)
                for index, path in enumerate(paths):
                    prediction = detached[index].argmax().item()
                    row = {"path": path, "target": y[index].item(), "prediction": prediction,
                           "confidence": probabilities[index, prediction].item()}
                    row.update({f"logit_{j}": value for j, value in enumerate(detached[index].tolist())})
                    row.update({f"probability_{j}": value for j, value in enumerate(probabilities[index].tolist())})
                    prediction_rows.append(row)
            predictions.extend(logits.detach().argmax(1).cpu().tolist())
            targets.extend(y.cpu().tolist())
    if not total:
        raise ValueError("Empty epoch")
    return {"loss": total_loss / loss_denominator, "accuracy": accuracy_score(targets, predictions) * 100,
            "macro_f1": f1_score(targets, predictions, labels=list(range(model._num_classes)) if hasattr(model, "_num_classes") else None,
                     average="macro", zero_division=0)}
