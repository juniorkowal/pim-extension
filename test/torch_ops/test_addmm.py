import pytest
import torch
import torch_pim

DEVICE = 'upmem'
RTOL = 1e-07

def test_addmm():
    M = torch.randn(2, 2)
    m1 = torch.randn(2, 2)
    m2 = torch.randn(2, 2)
    M_pim = M.upmem()
    m1_pim = m1.upmem()
    m2_pim = m2.upmem()
    out_cpu = torch.addmm(input=M, mat1=m1, mat2=m2)
    out_pim = torch.addmm(input=M_pim, mat1=m1_pim, mat2=m2_pim)
    out_pim = out_pim.cpu()
    assert torch.allclose(out_cpu, out_pim, RTOL) == True

def test_addmm_inplace():
    M = torch.randn(2, 2)
    m1 = torch.randn(2, 2)
    m2 = torch.randn(2, 2)
    M_pim = M.upmem()
    m1_pim = m1.upmem()
    m2_pim = m2.upmem()
    M.addmm_(m1, m2)
    M_pim.addmm_(m1_pim, m2_pim)
    M_pim = M_pim.cpu()
    assert torch.allclose(M, M_pim, RTOL) == True

def test_addmm_out():
    M = torch.randn(2, 2)
    m1 = torch.randn(2, 2)
    m2 = torch.randn(2, 2)
    M_pim = M.upmem()
    m1_pim = m1.upmem()
    m2_pim = m2.upmem()
    out_cpu = torch.empty_like(M)
    out_pim = out_cpu.upmem()
    torch.addmm(input=M, mat1=m1, mat2=m2, out=out_cpu)
    torch.addmm(input=M_pim, mat1=m1_pim, mat2=m2_pim, out=out_pim)
    out_pim = out_pim.cpu()
    assert torch.allclose(out_cpu, out_pim, RTOL) == True

def test_addmm_large():
    M = torch.randn(512, 512)
    m1 = torch.randn(512, 512)
    m2 = torch.randn(512, 512)
    M_pim = M.upmem()
    m1_pim = m1.upmem()
    m2_pim = m2.upmem()
    out_cpu = torch.addmm(input=M, mat1=m1, mat2=m2)
    out_pim = torch.addmm(input=M_pim, mat1=m1_pim, mat2=m2_pim)
    out_pim = out_pim.cpu()
    assert torch.allclose(out_cpu, out_pim, RTOL) == True
