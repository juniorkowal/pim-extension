import torch

@torch.library.register_fake("hpim::pim_mm")
def _(a, b): return torch.empty_like(a)

@torch.library.register_fake("hpim::pim_add")
def _(a, b): return torch.empty_like(a)

@torch.library.register_fake("hpim::pim_relu")
def _(a, out=None): return torch.empty_like(a)

mm = torch.ops.hpim.pim_mm
relu = torch.ops.hpim.pim_relu
add = torch.ops.hpim.pim_add
