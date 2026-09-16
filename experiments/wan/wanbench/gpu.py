"""Exploratory Wan operator measurement on B300; trusted candidate code only.

This is NOT the isolated final evaluator. No framework performance claim is made.
API sources: https://docs.pytorch.org/docs/2.12/generated/torch.compile.html
https://docs.pytorch.org/docs/2.12/profiler.html
https://docs.pytorch.org/docs/2.12/generated/torch.cuda.Event.html
Target runtime observed on AIS: PyTorch 2.12.0a0 NVIDIA build, CUDA 13.2.
"""
import importlib.util
import os
import platform
import statistics
import subprocess
import time

from .core import (digest, dump_new, load_json, object_digest, positive_int,
                   validate_device_selection)
from .tasks import build_reference, make_inputs, flatten_outputs, output_structure


def check_output(torch, fn, inputs, expected, atol, rtol):
    before = tuple(t.clone() for t in inputs)
    output = fn(*inputs)
    torch.cuda.synchronize()
    unchanged = all(torch.equal(a, b) for a, b in zip(before, inputs))
    if output_structure(torch, output) != output_structure(torch, expected):
        return {"passed": False, "reason": "output structure", "inputs_unchanged": unchanged}
    actuals, targets = flatten_outputs(torch, output), flatten_outputs(torch, expected)
    valid = bool(actuals) and all(a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
                                 for a, b in zip(actuals, targets))
    if not valid:
        return {"passed": False, "reason": "output shape/dtype/device", "inputs_unchanged": unchanged}
    finite = all(bool(torch.isfinite(o).all().item()) for o in actuals)
    errors = [(a.double() - b.double()).abs() for a, b in zip(actuals, targets)]
    max_abs = max(e.max().item() for e in errors) if finite else None
    passed = finite and unchanged and all(bool((e <= atol + rtol * b.double().abs()).all().item())
                                         for e, b in zip(errors, targets))
    return {"passed": passed, "finite": finite, "inputs_unchanged": unchanged,
            "max_abs_error": max_abs, "atol": atol, "rtol": rtol}


def run_smoke(args, root):
    positive_int(args.warmup, "warmup"); positive_int(args.samples, "samples")
    if args.samples < 2:
        raise ValueError("at least two samples required")
    if args.output.exists():
        raise FileExistsError(args.output)
    if (args.variant == "candidate") != (args.candidate is not None):
        raise ValueError("--candidate is required only with --variant candidate")
    validate_device_selection(load_json(root / "configs/device.json"),
                              os.environ.get("CUDA_VISIBLE_DEVICES"), args.device)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU substitution or fabricated timing")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    name = torch.cuda.get_device_name(device)
    if "B300" not in name:
        raise RuntimeError(f"this campaign requires B300, found {name}")
    task = next(t for t in load_json(root / "configs/tasks.json")["tasks"] if t["id"] == args.task)
    shape = task["cases"][args.case]
    dtype = getattr(torch, args.dtype)
    props = torch.cuda.get_device_properties(device)
    driver = subprocess.run(["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True, timeout=15)
    if driver.returncode:
        raise RuntimeError("could not record GPU identity and driver")
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "cuda": torch.version.cuda, "name": name, "device_index": args.device,
           "uuid": str(getattr(props, "uuid", "unknown")),
           "capability": list(torch.cuda.get_device_capability(device)),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "driver_inventory": driver.stdout.strip()}
    config = {"shape": shape, "dtype": args.dtype, "seed": args.seed,
              "task": args.task, "reference_source_sha256": digest(root / "wanbench/tasks.py"),
              "correctness_cases": "seeds 17/18/repeat relative to CLI seed, plus seed+2 shifted by +2 in first input",
              "distribution": "synthetic CPU randn; task-specific scaling; see tasks.py"}
    measurement = {"warmup": args.warmup, "samples": args.samples,
                   "timing": "CUDA events on current stream; candidate must join auxiliary work before returning",
                   "input_cache": "single-synthetic-input-hot-cache",
                   "matmul_tf32": False, "cudnn_tf32": False}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    candidate_hash = digest(args.candidate) if args.candidate else None
    result = {"schema_version": 1, "kind": "exploratory-gpu", "formal_ready": False,
              "variant": args.variant, "task": task["id"], "case": args.case,
              "task_hash": object_digest(task), "fixture_hash": object_digest(config),
              "environment": env, "environment_hash": object_digest(env),
              "measurement": measurement, "measurement_hash": object_digest(measurement),
              "candidate_sha256": candidate_hash, "profile": args.profile,
              "status": "not-run", "samples_ms": None, "whole_block_speedup": None}
    def inputs(seed):
        return make_inputs(torch, args.task, args.case, seed, dtype, device)
    base = build_reference(torch, args.task)
    if args.variant == "candidate":
        spec = importlib.util.spec_from_file_location("wanbench_candidate", args.candidate.resolve())
        if spec is None or spec.loader is None:
            raise ValueError("candidate must be an importable Python file")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = mod.run
    elif args.variant == "compile-default":
        fn = torch.compile(base, backend="inductor", mode="default", fullgraph=False)
    else:
        fn = base
    if not callable(fn):
        raise ValueError("candidate.run must be callable")
    with torch.inference_mode():
        checks = []
        for seed, stress in [(args.seed, False), (args.seed + 1, False), (args.seed, False), (args.seed + 2, True)]:
            values = inputs(seed)
            if stress:
                values[0].add_(2)
            expected = base(*values)
            if isinstance(expected, tuple):
                expected = tuple(t.clone() for t in expected)
            else:
                expected = expected.clone()
            start = time.perf_counter()
            verdict = check_output(torch, fn, values, expected, **task["smoke_tolerance"])
            verdict["seed"] = seed
            verdict["distribution"] = "first-input-shifted-plus-two" if stress else "random"
            verdict["call_wall_seconds"] = time.perf_counter() - start
            checks.append(verdict)
            del expected, values
            if not verdict["passed"]:
                break
        result["correctness"] = {"passed": all(x["passed"] for x in checks), "checks": checks,
                                 "scope": "public synthetic smoke inputs only"}
        if not result["correctness"]["passed"]:
            result["status"] = "fail"
            dump_new(args.output, result)
            return result
        values = inputs(args.seed)
        for _ in range(args.warmup):
            output = fn(*values)
        torch.cuda.synchronize()
        del output
        torch.cuda.reset_peak_memory_stats(device)
        if args.profile == "none":
            samples, walls = [], []
            for _ in range(args.samples):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start = time.perf_counter()
                a.record()
                output = fn(*values)
                b.record()
                b.synchronize()
                torch.cuda.synchronize()
                walls.append((time.perf_counter() - start) * 1000)
                samples.append(a.elapsed_time(b))
                del output
            result.update(samples_ms=samples, wall_samples_ms=walls,
                          p50_ms=statistics.median(samples))
        elif args.profile == "torch":
            from torch.profiler import profile, ProfilerActivity, record_function
            trace = args.output.with_suffix(".trace.json")
            if trace.exists():
                raise FileExistsError(trace)
            trace.parent.mkdir(parents=True, exist_ok=True)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for _ in range(5):
                    with record_function("wanbench/" + args.task):
                        output = fn(*values)
                    prof.step()
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(trace))
            result["trace"] = str(trace)
        else:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            try:
                for _ in range(5):
                    with torch.cuda.nvtx.range("wanbench/" + args.task):
                        output = fn(*values)
                torch.cuda.synchronize()
            finally:
                torch.cuda.cudart().cudaProfilerStop()
            result["capture_note"] = "Requires external nsys profile; no report existence is asserted."
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        result["status"] = "pass"
        dump_new(args.output, result)
    return result
