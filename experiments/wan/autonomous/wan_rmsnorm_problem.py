"""Custom Wan RMSNorm task; not an official KernelBench dataset problem."""
import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, x, weight):
        f = x.float()
        normalized = f * torch.rsqrt(f.square().mean(dim=-1, keepdim=True) + 1e-6)
        return normalized.to(x.dtype) * weight


def get_inputs():
    return [torch.randn(1, 7800, 1536, dtype=torch.bfloat16),
            torch.randn(1536, dtype=torch.bfloat16)]


def get_init_inputs():
    return []
