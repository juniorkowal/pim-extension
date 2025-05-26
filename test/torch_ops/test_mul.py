import pytest
import torch
import torch_pim

DEVICE = 'upmem'
RTOL = 1e-07

def test_mul_case1():
    x_left_cpu = torch.randn(4, 1)
    x_right_cpu = torch.randn(4)
    x_left_pim = x_left_cpu.upmem()
    x_right_pim = x_right_cpu.upmem()
    out_cpu = torch.mul(x_left_cpu, x_right_cpu)
    out_pim = torch.mul(x_left_pim, x_right_pim)
    out_pim = out_pim.cpu()
    assert torch.allclose(out_cpu, out_pim, RTOL) == True

def test_mul_case2():
    x_left_cpu = torch.randn(1, 4)
    x_right_cpu = torch.randn(4)
    x_left_pim = x_left_cpu.upmem()
    x_right_pim = x_right_cpu.upmem()
    out_cpu = torch.mul(x_left_cpu, x_right_cpu)
    out_pim = torch.mul(x_left_pim, x_right_pim)
    out_pim = out_pim.cpu()
    assert torch.allclose(out_cpu, out_pim, RTOL) == True

def test_mul_inplace():
    x_left_cpu = torch.randn(4, 1)
    x_right_cpu = torch.randn(4)
    x_left_pim = x_left_cpu.upmem()
    x_right_pim = x_right_cpu.upmem()
    x_left_cpu.mul_(x_right_cpu)
    x_left_pim.mul_(x_right_pim)
    x_left_pim = x_left_pim.cpu()
    assert torch.allclose(x_left_cpu, x_left_pim, RTOL) == True

def test_mul_out():
    x_left_cpu = torch.randn(4, 1)
    x_right_cpu = torch.randn(4)
    x_left_pim = x_left_cpu.upmem()
    x_right_pim = x_right_cpu.upmem()
    out_cpu = torch.empty(size=[4, 4], dtype=x_left_cpu.dtype)
    out_pim = out_cpu.upmem()
    torch.mul(x_left_cpu, x_right_cpu, out=out_cpu)
    torch.mul(x_left_pim, x_right_pim, out=out_pim)
    out_pim = out_pim.cpu()
    assert torch.allclose(out_cpu, out_pim, RTOL) == True

def test_mul_large():
    x_left_cpu = torch.randn(512, 1)
    x_right_cpu = torch.randn(512)
    x_left_pim = x_left_cpu.upmem()
    x_right_pim = x_right_cpu.upmem()
    out_cpu = torch.mul(x_left_cpu, x_right_cpu)
    out_pim = torch.mul(x_left_pim, x_right_pim)
    out_pim = out_pim.cpu()
    assert torch.allclose(out_cpu, out_pim, RTOL) == True