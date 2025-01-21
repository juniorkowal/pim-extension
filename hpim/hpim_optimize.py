import torch
import torch.nn as nn
from .layers import PIMLinear, PIMReLU


def optimize(model: nn.Module, layers: list = None, mode: str = 'replace'):
    if layers is None:
        layers = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'linear' in layers:
            if mode == 'replace':
                weight = module.weight
                bias = module.bias
                new_layer = PIMLinear(module.in_features, module.out_features, weight=weight, bias=bias)
                parent_module = model
                for part in name.split('.')[:-1]:
                    parent_module = getattr(parent_module, part)
                setattr(parent_module, name.split('.')[-1], new_layer)
            elif mode == 'patch':
                module.forward = PIMLinear(module.in_features, module.out_features).forward

        elif isinstance(module, nn.ReLU) and 'relu' in layers:
            if mode == 'replace':
                inplace = module.inplace
                new_layer = PIMReLU(inplace)
                parent_module = model
                for part in name.split('.')[:-1]:
                    parent_module = getattr(parent_module, part)
                setattr(parent_module, name.split('.')[-1], new_layer)
            elif mode == 'patch':
                module.forward = PIMReLU(module.inplace).forward

    return model
