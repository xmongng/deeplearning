"""Packed W4A32/W2A32 weights; inference unpacks once to FP32, not INT4 kernels."""
import math
import torch


def pack_weight(weight, bits):
    if bits not in (2, 4):
        raise ValueError('Only 2 or 4 bits supported')
    w = weight.detach().cpu().float().reshape(weight.shape[0], -1)
    if not torch.isfinite(w).all():
        raise ValueError('Non-finite weight')
    maximum = (1 << (bits - 1)) - 1
    scale = w.abs().amax(1).clamp_min(torch.finfo(torch.float32).tiny) / maximum
    codes = (torch.round(w / scale[:, None]).clamp(-maximum, maximum).to(torch.int16)
             + (1 << (bits - 1))).flatten().to(torch.uint8)
    per_byte = 8 // bits
    padding = (-codes.numel()) % per_byte
    if padding:
        codes = torch.cat((codes, torch.zeros(padding, dtype=torch.uint8)))
    packed = torch.zeros(codes.numel() // per_byte, dtype=torch.uint8)
    for i in range(per_byte):
        packed |= codes[i::per_byte] << (bits * i)
    return {'packed': packed, 'scale': scale, 'shape': list(weight.shape), 'bits': bits}


def unpack_weight(record):
    bits = record['bits']
    packed = record['packed']
    codes = torch.stack([(packed >> (bits * i)) & ((1 << bits) - 1)
                         for i in range(8 // bits)], dim=1).flatten()
    q = codes[:math.prod(record['shape'])].float() - (1 << (bits - 1))
    return (q.reshape(record['shape'][0], -1) * record['scale'][:, None]).reshape(record['shape'])


def pack_model(model, bits):
    names = {name + '.weight' if name else 'weight' for name, module in model.named_modules()
             if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))}
    state = model.state_dict()
    return {'format': 'waste-edge-packed-symmetric-v1', 'bits': bits,
            'packed_weights': {name: pack_weight(state[name], bits) for name in sorted(names)},
            'remaining_state': {name: value.detach().cpu() for name, value in state.items() if name not in names}}


def restore_model(model, payload):
    if payload['format'] != 'waste-edge-packed-symmetric-v1':
        raise ValueError('Unsupported packed model format')
    state = dict(payload['remaining_state'])
    state.update({name: unpack_weight(record) for name, record in payload['packed_weights'].items()})
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
