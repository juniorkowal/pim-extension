# import pytest
# import torch
# import torch_pim

# DEVICE = 'upmem'

# def test_opcheck_mm():
#     A = torch.randn(size=[2,2], device=DEVICE)
#     B = torch.randn(size=[2,2], device=DEVICE)
#     torch.library.opcheck(torch.ops.aten.mm.default, args=(A, B))