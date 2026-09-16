import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .core import (build_plan, compare_smoke, digest, dump_new, load_json,
                   object_digest, safe_id, verify_sources)
from .tasks import TASK_IDS

ROOT = Path(__file__).resolve().parents[1]


def probe_command(argv):
    if shutil.which(argv[0]) is None:
        return {"available": False}
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        return {"available": True, "returncode": p.returncode,
                "output": (p.stdout + p.stderr)[-12000:]}
    except subprocess.TimeoutExpired:
        return {"available": True, "error": "timeout"}


def doctor():
    # No credentials, complete environment dumps, process args, or cloud tokens.
    result = {"python": sys.version.split()[0], "platform": platform.platform(),
              "gpu": probe_command(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,driver_version", "--format=csv,noheader"]),
              "nsys": probe_command(["nsys", "--version"]),
              "nvcc": probe_command(["nvcc", "--version"]),
              "torch_installed": importlib.util.find_spec("torch") is not None}
    if result["torch_installed"]:
        result["torch"] = probe_command([sys.executable, "-c",
            "import torch,json; print(json.dumps({'version':torch.__version__,"
            "'cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),"
            "'compiled_arches':torch.cuda.get_arch_list() if torch.cuda.is_available() else []}))"])
    result["formal_ready"] = False
    result["note"] = "Inventory only; B300 compiler/kernel/profiler smoke tests are still required."
    return result


def prepare(root, framework_id, task_id, name):
    safe_id(name)
    frameworks = {x["id"]: x for x in load_json(ROOT / "configs/frameworks.lock.json")["frameworks"]}
    tasks = {x["id"]: x for x in load_json(ROOT / "configs/tasks.json")["tasks"]}
    if framework_id not in frameworks or task_id not in tasks:
        raise ValueError("unknown framework or task")
    target = Path(root) / name
    target.mkdir(parents=True, exist_ok=False)
    (target / "candidate").mkdir()
    task = tasks[task_id]
    dump_new(target / "task.json", task)
    dump_new(target / "trial.json", {"status": "prepared-not-started", "framework": frameworks[framework_id],
             "task_hash": object_digest(task), "kind": "bringup", "framework_launched": False,
             "candidate_generated": False, "formal_ready": False})
    (target / "TASK.md").write_text(
        f"# {name}\n\nFramework: {framework_id}\nTask: {task_id}\n\n"
        "Read task.json and the pinned upstream workflow. Keep its search mechanism.\n"
        "The candidate directory is intentionally empty. Do not copy optimized upstream answers.\n"
        "For T1, submit candidate/kernel.py exposing run(x, scale, shift). The function must return\n"
        "one output tensor and must not mutate inputs. All input-dependent work belongs in run.\n"
        "Other tasks are specification-only in v0. No framework is launched by prepare.\n"
        "No formal score until the external evaluator and runtime are frozen.\n", encoding="utf-8")
    return {"workspace": str(target.resolve()), "status": "prepared-not-started"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Wan agent benchmark preparation; no autonomous search launched")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("doctor"); p.add_argument("--output", type=Path)
    sub.add_parser("verify-sources")
    p = sub.add_parser("plan"); p.add_argument("--campaign", type=Path, default=ROOT / "configs/pilot.json"); p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("prepare"); p.add_argument("--framework", required=True); p.add_argument("--task", default="t1-norm-modulation"); p.add_argument("--name", required=True); p.add_argument("--root", type=Path, default=ROOT / "workspaces")
    p = sub.add_parser("compare-smoke"); p.add_argument("baseline", type=Path); p.add_argument("candidate", type=Path)
    p = sub.add_parser("smoke"); p.add_argument("--variant", choices=["eager", "compile-default", "candidate"], default="compile-default"); p.add_argument("--candidate", type=Path); p.add_argument("--case", choices=["debug", "medium", "long"], default="debug"); p.add_argument("--profile", choices=["none", "torch", "nsys"], default="none"); p.add_argument("--output", type=Path, required=True); p.add_argument("--seed", type=int, default=17); p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16"); p.add_argument("--warmup", type=int, default=20); p.add_argument("--samples", type=int, default=100); p.add_argument("--device", type=int, default=0)
    p.add_argument("--task", choices=TASK_IDS, default="t1-norm-modulation")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            result = doctor()
            if args.output:
                dump_new(args.output, result)
        elif args.command == "verify-sources":
            result = verify_sources(ROOT, load_json(ROOT / "sources/manifest.json"))
        elif args.command == "plan":
            fs = {x["id"]: x for x in load_json(ROOT / "configs/frameworks.lock.json")["frameworks"]}
            ts = {x["id"]: x for x in load_json(ROOT / "configs/tasks.json")["tasks"]}
            result = build_plan(load_json(args.campaign), fs, ts)
            dump_new(args.output, result)
            result = {"plan": str(args.output), "trials": len(result["trials"]), "execution_enabled": False}
        elif args.command == "prepare":
            result = prepare(args.root, args.framework, args.task, args.name)
        elif args.command == "compare-smoke":
            result = compare_smoke(load_json(args.baseline), load_json(args.candidate))
        elif args.command == "smoke":
            from .gpu import run_smoke
            result = run_smoke(args, ROOT)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 1 if result.get("ok") is False or result.get("status") == "fail" else 0
    except (ValueError, KeyError, FileExistsError, FileNotFoundError, RuntimeError, ImportError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
