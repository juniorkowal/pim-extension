import operator
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import GraphModule, Transformer, symbolic_trace
from torch.fx.node import Argument

from .. import ops


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
    

def trace_fx(model: nn.Module) -> torch.fx.GraphModule:
    gm = torch.fx.symbolic_trace(model)
    transformed: torch.nn.Module = DecomposeTransformer(gm).transform()
    return transformed