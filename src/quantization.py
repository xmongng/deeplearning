"""Backend-specific static W8A8 PT2E helpers shared by PTQ and QAT."""
import copy

import torch
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer, get_symmetric_quantization_config)
from torchao.quantization.pt2e import move_exported_model_to_eval
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e, prepare_qat_pt2e


def example(cfg, batch=1, device="cpu"):
    return torch.randn(batch, 3, cfg["preprocessing"]["crop"], cfg["preprocessing"]["crop"], device=device)


def prepare(model, cfg, qat=False, device="cpu"):
    model.to(device)
    model.train(qat)
    # Symbolic batch supports all training batches; deployment is re-exported at batch 1.
    inputs = (example(cfg, 2 if qat else 1, device),)
    dynamic = ({0: torch.export.Dim("batch", min=1, max=max(128, cfg["qat"]["batch_size"]))},) if qat else None
    captured = torch.export.export(model, inputs, dynamic_shapes=dynamic).module()
    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_qat=qat))
    return (prepare_qat_pt2e if qat else prepare_pt2e)(captured, quantizer)


def convert(model):
    result = convert_pt2e(copy.deepcopy(model))
    move_exported_model_to_eval(result)
    return result


def freeze_bn(model):
    for node in model.graph.nodes:
        if node.op == "call_function" and str(node.target) in {
            "aten._native_batch_norm_legit.default", "aten.cudnn_batch_norm.default",
            "aten.batch_norm.default", "aten.native_batch_norm.default"}:
            args = list(node.args)
            args[5] = False
            node.args = tuple(args)
    model.recompile()
