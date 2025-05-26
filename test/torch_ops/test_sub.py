import pytest
import torch
import torch_pim

DEVICE = 'upmem'
RTOL = 1e-07

def test_sub():
    x_left_cpu = torch.randn(2, 2)
    x_right_cpu = torch.randn(2, 2)
    x_left_pim = x_left_cpu.upmem()
    x_right_pim = x_right_cpu.upmem()
    out_cpu = torch.sub(x_left_cpu, x_right_cpu)
    out_pim = torch.sub(x_left_pim, x_right_pim)
    out_pim = out_pim.cpu()
    assert torch.allclose(out_cpu, out_pim, RTOL) == True

def test_sub_inplace():
    x_left_cpu = torch.randn(2, 2)
    x_right_cpu = torch.randn(2, 2)
    x_left_pim = x_left_cpu.upmem()
    x_right_pim = x_right_cpu.upmem()
    x_left_cpu.sub_(x_right_cpu)
    x_left_pim.sub_(x_right_pim)
    x_left_pim = x_left_pim.cpu()
    assert torch.allclose(x_left_cpu, x_left_pim, RTOL) == True

def test_sub_out():
    x_left_cpu = torch.randn(2, 2)
    x_right_cpu = torch.randn(2, 2)
    x_left_pim = x_left_cpu.upmem()
    x_right_pim = x_right_cpu.upmem()
    out_cpu = torch.empty_like(x_left_cpu)
    out_pim = out_cpu.upmem()
    torch.sub(x_left_cpu, x_right_cpu, out=out_cpu)
    torch.sub(x_left_pim, x_right_pim, out=out_pim)
    out_pim = out_pim.cpu()
    assert torch.allclose(out_cpu, out_pim, RTOL) == True

# # gives: Floating point exception (core dumped)
# def test_sub_empty():
#     m1 = torch.tensor([1.0], dtype=torch.float, device=DEVICE)
#     m2 = torch.tensor([], dtype=torch.float, device=DEVICE)
#     res = m1 + m2
#     res = res.cpu()
#     assert res.cpu().shape == m2.shape

def test_sub_scalar():
    x_left_cpu = torch.randn(2, 2)
    x_right_cpu = torch.randn(1)
    x_left_pim = x_left_cpu.upmem()
    x_right_pim = x_right_cpu.upmem()
    out_cpu = torch.sub(x_left_cpu, x_right_cpu)
    out_pim = torch.sub(x_left_pim, x_right_pim)
    out_pim = out_pim.cpu()
    assert torch.allclose(out_cpu, out_pim, RTOL) == True

def test_sub_large():
    x_left_cpu = torch.randn(1024, 1024)
    x_right_cpu = torch.randn(1024, 1024)
    x_left_pim = x_left_cpu.upmem()
    x_right_pim = x_right_cpu.upmem()
    out_cpu = torch.sub(x_left_cpu, x_right_cpu)
    out_pim = torch.sub(x_left_pim, x_right_pim)
    out_pim = out_pim.cpu()
    assert torch.allclose(out_cpu, out_pim, RTOL) == True

def test_sub_broadcasting():
    test_cases = [
        {"A_shape": (5, 4), "B_shape": (1,), "expected_shape": (5, 4)},
        {"A_shape": (5, 4), "B_shape": (4,), "expected_shape": (5, 4)},
        {"A_shape": (15, 3, 5), "B_shape": (15, 1, 5), "expected_shape": (15, 3, 5)},
        {"A_shape": (15, 3, 5), "B_shape": (3, 5), "expected_shape": (15, 3, 5)},
        {"A_shape": (15, 3, 5), "B_shape": (3, 1), "expected_shape": (15, 3, 5)},
        {"A_shape": (8, 1, 6, 1), "B_shape": (7, 1, 5), "expected_shape": (8, 7, 6, 5)},
        {"A_shape": (256, 256, 3), "B_shape": (3,), "expected_shape": (256, 256, 3)},
    ]

    for idx, test_case in enumerate(test_cases, start=1):
            x_left_cpu = torch.randn(test_case["A_shape"])
            x_right_cpu = torch.randn(test_case["B_shape"])
            x_left_pim = x_left_cpu.upmem()
            x_right_pim = x_right_cpu.upmem()
            out_cpu = torch.sub(x_left_cpu, x_right_cpu)
            out_pim = torch.sub(x_left_pim, x_right_pim)
            out_pim = out_pim.cpu()
            assert out_cpu.shape == out_pim.shape
            assert torch.allclose(out_cpu, out_pim, RTOL) == True
            