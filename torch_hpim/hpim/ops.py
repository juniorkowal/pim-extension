import torch

@torch.library.register_fake("torch_hpim::mm")
def _(a, b): return torch.empty_like(a)

@torch.library.register_fake("torch_hpim::add")
def _(a, b): return torch.empty_like(a)

@torch.library.register_fake("torch_hpim::relu")
def _(a, out=None): return torch.empty_like(a)

import torch_hpim._C

mm = torch.ops.torch_hpim.mm
relu = torch.ops.torch_hpim.relu
add = torch.ops.torch_hpim.add
# add = torch_hpim._C.pim_add

