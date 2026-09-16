"""Run bounded, sequential baseline measurements; never launches an agent."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from wanbench.core import digest, dump_new, load_json
from wanbench.tasks import TASK_IDS

ROOT = Path(__file__).resolve().parent


def run_bounded(cmd, logfile, env, seconds):
    with logfile.open("w") as log:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                             env=env, start_new_session=True)
        try:
            return p.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            return 124


def resolve_candidate(root, mapping, task):
    path = (root / mapping[task]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("candidate must stay inside experiment directory")
    generation = load_json(path.parent / "generation.json")
    if generation["task"] != task or digest(path) != generation["candidate_sha256"]:
        raise ValueError("candidate differs from recorded generation")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", choices=TASK_IDS, default=list(TASK_IDS))
    ap.add_argument("--cases", nargs="+", choices=["debug", "medium", "long"], default=["debug"])
    ap.add_argument("--variants", nargs="+", choices=["eager", "compile-default", "candidate"], default=["eager", "compile-default"])
    ap.add_argument("--candidate-map", type=Path)
    ap.add_argument("--reviewed-candidates", action="store_true", help="operator confirms source review; this is not a sandbox")
    ap.add_argument("--profiles", action="store_true")
    ap.add_argument("--timing", choices=["single-call", "cuda-graph"], default="single-call")
    ap.add_argument("--nsys-only", action="store_true", help="retry nsys capture without repeating valid measurements")
    args = ap.parse_args()
    if args.timing == "cuda-graph" and (args.profiles or args.nsys_only):
        raise ValueError("graph timing must be separate from ordinary-call profiles")
    candidates = {}
    if "candidate" in args.variants:
        if not args.reviewed_candidates or args.candidate_map is None:
            raise ValueError("candidate execution requires source review and a candidate map")
        mapping = load_json(args.candidate_map)
        candidates = {task: resolve_candidate(ROOT, mapping, task) for task in args.tasks}
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = load_json(ROOT / "configs/device.json")["uuid"]
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    temporary = out / "tmp"
    temporary.mkdir()
    env["TMPDIR"] = str(temporary)
    plan = {"status": "running", "kind": "synthetic-preflight", "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "gpu_uuid": env["CUDA_VISIBLE_DEVICES"], "variants": args.variants, "tasks": args.tasks, "cases": args.cases, "results": []}
    # This coordinates our jobs; it is not a device reservation against other users.
    with (ROOT / "gpu3.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        def occupied():
            probe = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=15, check=True)
            return [line.strip() for line in probe.stdout.splitlines() if env["CUDA_VISIBLE_DEVICES"] in line]
        for task in args.tasks:
            for case in args.cases:
                variants = [(variant, "none") for variant in args.variants]
                if args.profiles:
                    variants += [(variant, profile) for variant in args.variants if variant != "eager" for profile in ("torch", "nsys")]
                if args.nsys_only:
                    variants = [(variant, "nsys") for variant in args.variants if variant != "eager"]
                for variant, profile in variants:
                    name = f"{task}-{case}-{variant}-{profile}"
                    target = out / (name + ".json")
                    cmd = [sys.executable, "-m", "wanbench", "smoke", "--task", task,
                           "--case", case, "--variant", variant, "--profile", profile,
                           "--samples", "50", "--warmup", "10", "--timing", args.timing,
                           "--output", str(target)]
                    if variant == "candidate":
                        candidate = resolve_candidate(ROOT, mapping, task)
                        cmd += ["--candidate", str(candidate)]
                    if profile == "nsys":
                        cmd = ["nsys", "profile", "--trace=cuda,nvtx,osrt", "--sample=none",
                               "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                               "--output=" + str(out / name)] + cmd
                    before = occupied()
                    if before:
                        plan.update(status="blocked-gpu-busy", occupying_processes=before)
                        dump_new(out / "summary.json", plan)
                        print("selected GPU has an existing compute process; leave it untouched", flush=True)
                        return 3
                    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    rc = run_bounded(cmd, out / (name + ".log"), env, 420)
                    after = occupied()
                    row = {"started": started, "ended": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           "occupancy_before": before, "occupancy_after": after, "contention_detected": bool(after),
                           "task": task, "case": case, "variant": variant, "profile": profile,
                           "returncode": rc, "status": "missing-artifact", "artifact": str(target) if target.exists() else None}
                    if target.exists():
                        data = load_json(target)
                        row.update(status=data["status"], p50_ms=data.get("p50_ms"),
                                   max_errors=[x.get("max_abs_error") for x in data["correctness"]["checks"]])
                    if profile == "nsys":
                        row["nsys_report_exists"] = (out / (name + ".nsys-rep")).is_file()
                    row["successful"] = (rc == 0 and row["status"] == "pass" and
                                         (profile != "nsys" or row["nsys_report_exists"]))
                    plan["results"].append(row)
                    # Snapshot for progress only, raw measurements stay immutable.
                    (out / "progress.json").write_text(json.dumps(plan, indent=2) + "\n")
                    print(json.dumps(row), flush=True)
                    if after:
                        plan["status"] = "blocked-gpu-busy"
                        dump_new(out / "summary.json", plan)
                        print("other process detected after measurement; timings need revalidation", flush=True)
                        return 3
                    if rc == 124:
                        raise RuntimeError("GPU subprocess timed out; stop preflight and inspect device before continuing")
    plan["status"] = "complete" if all(r["successful"] for r in plan["results"]) else "completed-with-failures"
    dump_new(out / "summary.json", plan)
    return 0 if plan["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
