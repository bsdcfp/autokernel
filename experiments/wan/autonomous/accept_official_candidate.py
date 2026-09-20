"""Post-run acceptance against frozen references and default torch.compile.

This evaluator never supplies feedback to the optimization Agent.
"""
import argparse
# Reserve the stdlib module before the frozen repository adds its own profile.py.
import cProfile
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def checked(fn, inputs, expected, tol):
    before = {k:v.clone() for k,v in inputs.items() if isinstance(v, torch.Tensor)}
    out = fn(**inputs)
    torch.cuda.synchronize()
    valid = (isinstance(out, torch.Tensor) and out.shape == expected.shape
             and out.dtype == expected.dtype and out.device == expected.device)
    finite = bool(valid and torch.isfinite(out).all())
    same_inputs = all(torch.equal(inputs[k], v) for k,v in before.items())
    match = bool(finite and torch.allclose(out.float(), expected.float(), **tol))
    error = float((out.float()-expected.float()).abs().max()) if valid else None
    return dict(pass_=match and same_inputs, output_contract=valid, finite=finite,
                inputs_unchanged=same_inputs, max_abs_error=error)


def timer(fn, count=32):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(count):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000 / count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--reference-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--profile", choices=["none", "torch", "nsys"], default="none")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.reference_root))
    bench = load("frozen_bench", args.reference_root / "bench.py")
    candidate = load("external_candidate", args.candidate)
    kind = candidate.KERNEL_TYPE
    cfg = bench.KERNEL_CONFIGS[kind]
    gen, ref = cfg["input_generator"], cfg["reference_fn"]
    record = dict(candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                  kernel_type=kind, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                  source="coordinator post-run acceptance", correctness=[], performance=[],
                  baseline="torch.compile(reference, mode=default)",
                  timing="CUDA events; 15 alternating paired rounds; 32 ordinary calls per sample; warm reused inputs; no CUDA Graph",
                  profile=args.profile,
                  float32_matmul_precision=torch.get_float32_matmul_precision(),
                  allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  allow_fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
    save = lambda: (args.output / "result.json").write_text(json.dumps(record, indent=2))
    sizes = list(cfg["test_sizes"]) + list(cfg.get("edge_sizes", []))
    primary = next((x for x in cfg["test_sizes"] if x[0] == "large"), cfg["test_sizes"][-1])
    # Actual matrix layouts implied by the unmodified compact LlamaModel's Linear layers.
    model_sizes = [("model_"+n, {"M":512,"N":n_out,"K":k}) for n,n_out,k in
                   [("q_o",768,768),("k_v",256,768),("gate_up",2048,768),
                    ("down",768,2048),("logits",32000,768)]] if kind == "matmul" else []
    def inputs_for(label, size, dtype, seed):
        inputs = gen(size, dtype, "cuda", seed=seed)
        if label.startswith("model_"):
            inputs["B"] = inputs["B"].T.contiguous().T
        return inputs
    with torch.inference_mode():
        if args.profile == "none":
            for label, size in sizes + model_sizes:
                for dtype in cfg["test_dtypes"]:
                    for seed in (913, 1729):
                        row = dict(case=label, size=size, dtype=str(dtype), seed=seed)
                        try:
                            inputs = inputs_for(label, size, dtype, seed)
                            expected = ref(inputs)
                            row.update(checked(candidate.kernel_fn, inputs, expected, cfg["tolerances"][dtype]))
                        except Exception as e:
                            row.update(pass_=False, error=type(e).__name__+": "+str(e))
                        record["correctness"].append(row)
                        save()
            record["all_correct"] = all(x["pass_"] for x in record["correctness"])
            save()
            if not record["all_correct"]:
                record["performance_status"] = "not scored: correctness failed"
                save()
                print(json.dumps({"all_correct":False,"failed":sum(not x["pass_"] for x in record["correctness"])}))
                return
        compiled = torch.compile(ref, mode="default")
        for label, size in ([primary] if args.profile != "none" else [primary] + model_sizes):
            dtype = torch.float16
            inputs = inputs_for(label, size, dtype, 2047)
            expected = ref(inputs)
            c = compiled(inputs)
            if not torch.allclose(c.float(), expected.float(), **cfg["tolerances"][dtype]):
                raise RuntimeError("Default compile baseline differs from frozen reference")
            check = checked(candidate.kernel_fn, inputs, expected, cfg["tolerances"][dtype])
            if not check["pass_"]:
                raise RuntimeError("Candidate failed fresh performance input")
            funcs = {"compile":lambda:compiled(inputs), "candidate":lambda:candidate.kernel_fn(**inputs)}
            for fn in funcs.values():
                for _ in range(10):fn()
            torch.cuda.synchronize()
            if args.profile == "torch":
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
                    for name, fn in funcs.items():
                        with torch.profiler.record_function(name):
                            for _ in range(10):fn()
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(args.output / "trace.json"))
                (args.output / "profiler.txt").write_text(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
            elif args.profile == "nsys":
                torch.cuda.cudart().cudaProfilerStart()
                for name, fn in funcs.items():
                    with torch.cuda.nvtx.range(name):
                        for _ in range(10):fn()
                    torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
            else:
                pairs = []
                for r in range(15):
                    pair = {name:timer(funcs[name]) for name in (("compile","candidate") if r%2==0 else ("candidate","compile"))}
                    pairs.append(pair)
                record["performance"].append(dict(case=label, size=size, dtype=str(dtype),
                    compile_us=statistics.median(x["compile"] for x in pairs),
                    candidate_us=statistics.median(x["candidate"] for x in pairs),
                    paired_speedup=statistics.median(x["compile"]/x["candidate"] for x in pairs),
                    logical_flops=cfg["flops_fn"](size), logical_bytes=cfg["bytes_fn"](size,dtype), samples=pairs))
            save()
    print(json.dumps({"all_correct":record.get("all_correct"),"performance":record["performance"]},indent=2))


if __name__ == "__main__":
    main()
