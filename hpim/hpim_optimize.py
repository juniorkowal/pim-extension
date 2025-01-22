import torch
import torch.nn as nn
from collections import defaultdict

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

def optimize(model: nn.Module, layers: list = None):
    """
    Optimizes a model by replacing specified operations/layers with PIM versions.
    
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
