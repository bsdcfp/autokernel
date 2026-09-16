"""Synthetic operator fixtures matching the pinned Wan2.1 semantics.

Semantic source (Alibaba Wan Team, 2024-2025):
https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py
Fixtures are synthetic. These functions are not whole-block adapters.
"""

GRIDS = {"debug": (1, 16, 16), "medium": (5, 30, 52), "long": (21, 30, 52)}
TASK_IDS = ("t1-norm-modulation", "wan-gated-residual", "wan-rmsnorm",
            "wan-rope3d", "t2-qknorm-rope", "wan-linear-gelu")


def norm_modulation(torch, x, scale, shift):
    normalized = torch.nn.functional.layer_norm(x.float(), (x.shape[-1],), eps=1e-6)
    return normalized.to(x.dtype).float() * (1 + scale) + shift


def rmsnorm(torch, x, weight):
    f = x.float()
    normalized = f * torch.rsqrt(f.square().mean(dim=-1, keepdim=True) + 1e-6)
    return normalized.to(x.dtype) * weight


def rope3d(torch, x, grid_sizes, freqs):
    heads, complex_dim = x.shape[2], x.shape[3] // 2
    widths = [complex_dim - 2 * (complex_dim // 3), complex_dim // 3, complex_dim // 3]
    ft, fh, fw = freqs.split(widths, dim=1)
    outputs = []
    for i, (t, h, w) in enumerate(grid_sizes.tolist()):
        length = t * h * w
        z = torch.view_as_complex(x[i, :length].double().reshape(length, heads, -1, 2))
        multiplier = torch.cat([
            ft[:t].view(t, 1, 1, -1).expand(t, h, w, -1),
            fh[:h].view(1, h, 1, -1).expand(t, h, w, -1),
            fw[:w].view(1, 1, w, -1).expand(t, h, w, -1),
        ], dim=-1).reshape(length, 1, -1)
        rotated = torch.view_as_real(z * multiplier).flatten(2)
        outputs.append(torch.cat([rotated, x[i, length:]], dim=0))
    return torch.stack(outputs).float()


def build_reference(torch, task_id):
    if task_id == "t1-norm-modulation":
        return lambda x, scale, shift: norm_modulation(torch, x, scale, shift)
    if task_id == "wan-gated-residual":
        return lambda x, y, gate: x + y * gate
    if task_id == "wan-rmsnorm":
        return lambda x, weight: rmsnorm(torch, x, weight)
    if task_id == "wan-rope3d":
        return lambda x, grid_sizes, freqs: rope3d(torch, x, grid_sizes, freqs)
    if task_id == "t2-qknorm-rope":
        def fused_reference(q, k, q_weight, k_weight, grid_sizes, freqs):
            qn = rmsnorm(torch, q, q_weight).reshape(q.shape[0], q.shape[1], 12, 128)
            kn = rmsnorm(torch, k, k_weight).reshape(k.shape[0], k.shape[1], 12, 128)
            return rope3d(torch, qn, grid_sizes, freqs), rope3d(torch, kn, grid_sizes, freqs)
        return fused_reference
    if task_id == "wan-linear-gelu":
        return lambda x, weight, bias: torch.nn.functional.gelu(
            torch.nn.functional.linear(x, weight, bias), approximate="tanh")
    raise ValueError(f"unknown executable task: {task_id}")


def make_inputs(torch, task_id, case, seed, dtype, device):
    if task_id not in TASK_IDS:
        raise ValueError(f"unknown executable task: {task_id}")
    grid = GRIDS[case]
    sequence = grid[0] * grid[1] * grid[2]
    shape, modshape = (1, sequence, 1536), (1, 1, 1536)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def rand(size, kind=dtype, scale=1.0):
        return (torch.randn(size, generator=generator) * scale).to(device=device, dtype=kind)

    def rope_metadata():
        # Match WanModel.__init__: three independently parameterized frequency bands.
        parts = []
        for width in [44, 42, 42]:
            exponent = torch.arange(0, width, 2, dtype=torch.float64) / width
            angles = torch.outer(torch.arange(1024, dtype=torch.float64),
                                 1.0 / torch.pow(10000.0, exponent))
            parts.append(torch.polar(torch.ones_like(angles), angles))
        return torch.tensor([grid], dtype=torch.int64), torch.cat(parts, dim=1).to(device)

    if task_id == "t1-norm-modulation":
        return rand(shape), rand(modshape, torch.float32, 0.1), rand(modshape, torch.float32, 0.1)
    if task_id == "wan-gated-residual":
        return rand(shape), rand(shape), rand(modshape, torch.float32, 0.1)
    if task_id == "wan-rmsnorm":
        return rand(shape), rand((1536,))
    if task_id == "wan-rope3d":
        x = rand((1, sequence, 12, 128))
        return x, *rope_metadata()
    if task_id == "t2-qknorm-rope":
        return rand(shape), rand(shape), rand((1536,)), rand((1536,)), *rope_metadata()
    # Fan-in scaling gives a representative activation range; weights remain synthetic.
    return rand(shape), rand((8960, 1536), scale=1536 ** -0.5), rand((8960,), scale=0.01)


def flatten_outputs(torch, output):
    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, (tuple, list)):
        flattened = []
        for item in output:
            flattened.extend(flatten_outputs(torch, item))
        return flattened
    raise ValueError("output must contain tensors only")


def output_structure(torch, output):
    if isinstance(output, torch.Tensor):
        return "tensor"
    if isinstance(output, (tuple, list)):
        return [type(output).__name__, [output_structure(torch, x) for x in output]]
    return "invalid"
