import torch.nn as nn
from .methods import switch_layers, trace_fx, pim_compile


class DeprecationError(Exception):
    pass

def optimize(model: nn.Module, mode: str = 'layers', *args, **kwargs):
    raise DeprecationError(
        "The 'optimize' function is currently depreacted. "
        "To use the UPMEM extension, please move your input and model to the 'upmem' device "
        "using .to('upmem') instead.\n\n"
        "Example usage:\n"
        "    model = model.to('upmem')\n"
        "    input = input.to('upmem')\n\n"
    ) 
    assert mode in ['layers', 'fx', 'compile']
    if mode == 'layers':
        return switch_layers(model)
    elif mode == 'fx':
        return trace_fx(model)
    elif mode == 'compile':
        return pim_compile(model)