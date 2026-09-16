"""Differential precision diagnostics; never a speed-scoring candidate.

Keep original references, candidates and tolerances immutable. The no-FMA
wrapper is an explicitly labelled diagnostic ablation of the reviewed source.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess

from wanbench.core import digest, dump_new, load_json, validate_device_selection
from wanbench.tasks import build_reference, make_inputs, rmsnorm, rope3d, flatten_outputs
from preflight import ROOT, resolve_candidate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['wan-linear-gelu', 't2-qknorm-rope'], required=True)
    ap.add_argument('--case', choices=['debug', 'medium', 'long'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    validate_device_selection(load_json(ROOT / 'configs/device.json'), os.environ.get('CUDA_VISIBLE_DEVICES'), 0)
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from wanbench.gpu import check_output
    torch.cuda.set_device(0)
    if 'B300' not in torch.cuda.get_device_name(0):
        raise RuntimeError('requires B300')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tolerance = {'atol': .01, 'rtol': .01}
    base = build_reference(torch, args.task)
    report = {'kind': 'precision-diagnostic', 'task': args.task, 'case': args.case,
              'torch': torch.__version__, 'reference_sha256': digest(ROOT / 'wanbench/tasks.py'),
              'diagnostic_sha256': digest(Path(__file__)), 'results': [], 'performance_claim': None}

    def compare(label, fn, values, target, seed):
        verdict = check_output(torch, fn, values, target, **tolerance)
        actual = fn(*values)
        aa, bb = flatten_outputs(torch, actual), flatten_outputs(torch, target)
        verdict['unequal_elements'] = sum(int((a != b).sum().item()) for a, b in zip(aa, bb))
        verdict['total_elements'] = sum(b.numel() for b in bb)
        verdict.update(label=label, seed=seed)
        report['results'].append(verdict)
        print(label, seed, 'PASS' if verdict['passed'] else 'FAIL',
              'bad=', verdict.get('failed_elements'), 'unequal=', verdict['unequal_elements'],
              'max=', verdict.get('max_abs_error'), flush=True)
        return actual

    compiled = {name: torch.compile(base, backend='inductor', fullgraph=False,
                                   options={'emulate_precision_casts': enabled})
                for name, enabled in [('compile-default', False), ('compile-preserve-casts', True)]}
    candidate = no_fma = None
    if args.task == 't2-qknorm-rope':
        path = resolve_candidate(ROOT, load_json(ROOT / 'configs/candidates-r02.json'), args.task)
        report['candidate_sha256'] = digest(path)
        def module(name):
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            return mod
        original = module('original_candidate')
        candidate = original.run
        ablation = module('diagnostic_no_fma')
        class NoFmaKernel:
            def __init__(self, kernel): self.kernel = kernel
            def __getitem__(self, grid):
                launch = self.kernel[grid]
                def run(*values, **meta):
                    meta['enable_fp_fusion'] = False
                    return launch(*values, **meta)
                return run
        ablation._rmsnorm_rope_kernel = NoFmaKernel(ablation._rmsnorm_rope_kernel)
        no_fma = ablation.run
        report['ablation'] = 'same generated source; kernel launch enable_fp_fusion=False; not Agent-generated fix'

    with torch.inference_mode():
        for seed, stress in [(17, False), (18, False), (17, False), (19, True)]:
            values = make_inputs(torch, args.task, args.case, seed, torch.bfloat16, torch.device('cuda:0'))
            if stress: values[0].add_(2)
            expected = base(*values)
            for label, fn in compiled.items():
                compare(label + ('-shifted' if stress else ''), fn, values, expected, seed)
            if candidate:
                compare('candidate-original' + ('-shifted' if stress else ''), candidate, values, expected, seed)
                compare('candidate-no-fma-ablation' + ('-shifted' if stress else ''), no_fma, values, expected, seed)
            del values, expected

        values = make_inputs(torch, args.task, args.case, 17, torch.bfloat16, torch.device('cuda:0'))
        if args.task == 'wan-linear-gelu':
            x, weight, bias = values
            linear = lambda x, w, b: torch.nn.functional.linear(x, w, b)
            gelu = lambda z: torch.nn.functional.gelu(z, approximate='tanh')
            z = linear(*values)
            zc = compare('stage-linear-default', torch.compile(linear), values, z, 17)
            compare('stage-gelu-default-on-eager-linear', torch.compile(gelu), (z,), gelu(z), 17)
            staged = torch.compile(gelu)(zc)
            compare('stage-separated-linear-gelu', lambda: staged, (), gelu(z), 17)
            # Compare the whole graph with a model that intentionally omits the
            # final linear BF16 rounding. Diagnostic only, never the reference.
            zfp = linear(x.float(), weight.float(), bias.float())
            unrounded = gelu(zfp).bfloat16()
            compare('hypothesis-unrounded-linear-vs-full-default', lambda: unrounded, (), compiled['compile-default'](*values), 17)
            if args.case == 'debug':
                error = (compiled['compile-default'](*values).float() - gelu(z).float()).abs()
                flat = int(error.flatten().argmax().item())
                row, column = divmod(flat, weight.shape[0])
                dot = (x.reshape(-1, x.shape[-1])[row].double() * weight[column].double()).sum() + bias[column].double()
                report['ffn_counterexample'] = {'row': row, 'column': column, 'dot_fp64': dot.item(),
                    'eager_linear': z.reshape(-1, weight.shape[0])[row, column].item(),
                    'compiled_linear': zc.reshape(-1, weight.shape[0])[row, column].item(),
                    'eager_gelu': gelu(z).reshape(-1, weight.shape[0])[row, column].item(),
                    'full_compiled_gelu': compiled['compile-default'](*values).reshape(-1, weight.shape[0])[row, column].item(),
                    'gelu_after_dot_bf16': gelu(dot.bfloat16()).item(),
                    'gelu_before_dot_bf16': gelu(dot).bfloat16().item()}
        else:
            q, k, qw, kw, grid, freqs = values
            identity = torch.ones_like(freqs)
            identity_values = (q, k, qw, kw, grid, identity)
            identity_target = base(*identity_values)
            compare('stage-candidate-identity-rope', candidate, identity_values, identity_target, 17)
            ones_values = (q, k, torch.ones_like(qw), torch.ones_like(kw), grid, identity)
            ones_target = base(*ones_values)
            normalized = compare('stage-candidate-unweighted-norm', candidate, ones_values, ones_target, 17)
            compare('stage-no-fma-unweighted-norm', no_fma, ones_values, ones_target, 17)
            # A FP64 diagnostic reduction identifies which BF16 rounding is
            # closer to a higher-precision norm, without changing the reference.
            for side, x, got, wanted in zip(['q','k'], [q,k], normalized, ones_target):
                actual = got.reshape_as(x)
                target = wanted.reshape_as(x)
                diffs = (actual != target).reshape(-1)
                if not bool(diffs.any().item()): continue
                index = int(diffs.nonzero()[0].item())
                row, col = divmod(index, x.shape[-1])
                xr = x.reshape(-1, x.shape[-1])[row]
                mean64 = xr.double().square().mean()
                inv64 = torch.rsqrt(mean64 + 1e-6)
                norm64 = xr[col].double() * inv64
                report.setdefault('norm_counterexamples', []).append({
                    'side': side, 'row': row, 'column': col, 'input': xr[col].item(),
                    'mean_square_fp64': mean64.item(),
                    'mean_square_eager_fp32': xr.float().square().mean().item(),
                    'normalized_fp64': norm64.item(), 'fp64_rounded_bf16': norm64.bfloat16().item(),
                    'candidate_bf16': actual.reshape(-1)[index].item(),
                    'eager_bf16': target.reshape(-1)[index].item()})
    dump_new(args.output, report)


if __name__ == '__main__':
    main()
