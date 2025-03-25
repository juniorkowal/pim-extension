import operator
from collections import defaultdict
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import GraphModule, Transformer, symbolic_trace
from torch.fx.node import Argument

from . import ops

from .layers import PIMLinear, PIMReLU

LAYER_REGISTRY = defaultdict(dict) # defaultdict to avoid missing key errors

def register_pim_layer(layer_type, pim_layer_class, **kwargs):
    """
    Register a custom PIM layer for a given layer type.
    """
    LAYER_REGISTRY[layer_type] = {
        'pim_layer': pim_layer_class,
        'params': kwargs # maybe not needed
    }

register_pim_layer(nn.Linear, PIMLinear)
register_pim_layer(nn.ReLU, PIMReLU)

def get_init_params(module):
    """
    Gets the parameters from the __init__ method of the module, excluding methods.
    """
    init_params = {}

    for param_name, param_value in module.named_parameters():
        init_params[param_name] = param_value

    for param_name, param_value in module.__dict__.items():
        if callable(param_value) or param_name.startswith('_'):
            continue
        init_params[param_name] = param_value

    return init_params

def switch_layers(model: nn.Module, layers: list = None):
    """
    Replaces pecified layers with PIM versions.
    
    Args:
    - model (nn.Module): The input model to optimize.
    - layers (str list): List of layer types to replace. If None, all layers are replaced.

    Returns:
    - nn.Module: The optimized model with PIM operations.
    """
    if layers is None:
        layers = [op.__name__.lower() for op in LAYER_REGISTRY.keys()]

    for name, module in model.named_modules():
        if isinstance(module, tuple(LAYER_REGISTRY.keys())) and type(module).__name__.lower() in layers:
            registered_layer = LAYER_REGISTRY[type(module)]

            pim_layer_class = registered_layer['pim_layer']
            params = registered_layer['params']

            init_params = get_init_params(module)
            new_layer = pim_layer_class(**init_params, **params)

            parent_module = model
            for part in name.split('.')[:-1]:
                parent_module = getattr(parent_module, part)
            
            setattr(parent_module, name.split('.')[-1], new_layer)

    return model


def relu_decomposition(x):
    return ops.relu(x)

def linear_decomposition(x, weight, bias):
    return ops.add(ops.mm(x, weight.t().contiguous()), bias)

def add(input, other):
    return ops.add(input, other)

def mm(input, other):
    return ops.mm(other, input)


decomposition_rules = {
    F.relu: relu_decomposition,
    nn.ReLU: relu_decomposition,
    nn.Linear: linear_decomposition,
    F.linear: linear_decomposition,
    "add": add,
    torch.add: add,
    operator.add: add,    
    "mm": mm,
    torch.mm: mm,
    operator.mul: mm,
}


class DecomposeTransformer(Transformer):
    def call_function(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if target in decomposition_rules:
            return decomposition_rules[target](*args, **kwargs)
        return super().call_function(target, args, kwargs)

    def call_module(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        module = self.tracer.root.get_submodule(target)
        module_type = type(module)
        if module_type in decomposition_rules:
            module_params = dict(module.named_parameters()) # Extract the module's parameters dynamically
            all_args = args + tuple(module_params.values()) # Combine args and kwargs with module parameters
            return decomposition_rules[module_type](*all_args, **kwargs)
        return super().call_module(target, args, kwargs)

def optimize(model: nn.Module, layers: list = None, mode: str = 'fx'):
    assert mode in ['layers', 'fx']
    if mode == 'fx':
        gm = torch.fx.symbolic_trace(model)
        transformed: torch.nn.Module = DecomposeTransformer(gm).transform()
        return transformed
    elif mode == 'layers':
        return switch_layers(model, layers)
