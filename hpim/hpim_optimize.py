import torch
import torch.nn as nn
from .layers import PIMLinear, PIMReLU


def optimize(model: nn.Module, operations: list = None):
    if operations is None:
        operations = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'linear' in operations:
            weight = module.weight
            bias = module.bias
            new_layer = PIMLinear(module.in_features, module.out_features, weight=weight, bias=bias)
            parent_module = model
            for part in name.split('.')[:-1]:
                parent_module = getattr(parent_module, part)
            setattr(parent_module, name.split('.')[-1], new_layer)

        elif isinstance(module, nn.ReLU) and 'relu' in operations:
            inplace = module.inplace
            new_layer = PIMReLU(inplace)
            parent_module = model
            for part in name.split('.')[:-1]:
                parent_module = getattr(parent_module, part)
            setattr(parent_module, name.split('.')[-1], new_layer)

    return model
