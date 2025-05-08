import torch

@torch.library.register_fake("torch_pim::mm")
def _(a, b): return torch.empty_like(a)

@torch.library.register_fake("torch_pim::add")
def _(a, b): return torch.empty_like(a)

@torch.library.register_fake("torch_pim::relu")
def _(a, out=None): return torch.empty_like(a)

import torch_pim._C

mm = torch.ops.torch_pim.mm
relu = torch.ops.torch_pim.relu
add = torch.ops.torch_pim.add
# add = torch_pim._C.pim_add

