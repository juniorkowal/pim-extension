import torch.nn as nn
from .methods import switch_layers, trace_fx, pim_compile


def optimize(model: nn.Module, mode: str = 'layers'):
    assert mode in ['layers', 'fx', 'compile']
    if mode == 'layers':
        return switch_layers(model)
    elif mode == 'fx':
        return trace_fx(model)
    elif mode == 'compile':
        return pim_compile(model)