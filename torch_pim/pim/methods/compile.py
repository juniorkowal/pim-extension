from numbers import Number
from typing import Any, Dict, Tuple

import torch
import torch._dynamo
import torch.nn as nn
from torch import nn
from torch._functorch.aot_autograd import aot_module_simplified
from torch._inductor.decomposition import decompositions #as default_decompositions
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import GraphModule, Transformer, symbolic_trace
from torch.fx.node import Argument

# import torch_hpim
from .fx import DecomposeModel
from .. import ops

def is_numeric(*args) -> bool:
    return any(isinstance(x, Number) for x in args)

def add(input, other, *args,**kwargs):
    if is_numeric(input, other):
        return torch.add(input, other)
    return ops.add(input.contiguous(), other.contiguous())

def relu(input, inplace=False, *args, **kwargs):
    if is_numeric(input):
        return torch.relu(input)
    out = torch.zeros_like(input)
    return ops.relu(input.contiguous(), out)

def mm(input, other, *args,**kwargs):
    if is_numeric(input, other):
        return torch.mm(input, other)
    return ops.mm(input.contiguous(), other.contiguous())


decomposition_rules = {
    torch.ops.aten.add.Tensor: add,
    torch.ops.prims.add.default: add,
    torch.ops.aten.mm.default: mm,
    torch.ops.aten.relu.default: relu,
}

# decompositions = default_decompositions.copy()
decompositions = torch._decomp.get_decompositions(decompositions)

def pim_backend(gm, sample_inputs):
    def pim_compiler(gm, sample_inputs):
        decomposer = DecomposeModel(module=gm,
                decomposition_rules=decomposition_rules)
        gm = decomposer.transform()
        print("Decomposed fx Graph in Aten IR:")
        print(gm.graph)
        gm.recompile()
        return gm

    return aot_module_simplified(
        gm,
        sample_inputs,
        decompositions=decompositions,
        fw_compiler=pim_compiler,
    )


def pim_compile(model = nn.Module) -> torch.fx.GraphModule:
    return torch.compile(model, backend=pim_backend, dynamic=True)

