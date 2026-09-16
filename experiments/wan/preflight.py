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

from wanbench.core import dump_new, load_json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", choices=TASK_IDS, default=list(TASK_IDS))
    ap.add_argument("--cases", nargs="+", choices=["debug", "medium", "long"], default=["debug"])
    ap.add_argument("--profiles", action="store_true")
    ap.add_argument("--nsys-only", action="store_true", help="retry nsys capture without repeating valid measurements")
    args = ap.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = load_json(ROOT / "configs/device.json")["uuid"]
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    temporary = out / "tmp"
    temporary.mkdir()
    env["TMPDIR"] = str(temporary)
    plan = {"status": "running", "kind": "synthetic-preflight", "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "gpu_uuid": env["CUDA_VISIBLE_DEVICES"], "tasks": args.tasks, "cases": args.cases, "results": []}
    # This coordinates our jobs; it is not a device reservation against other users.
    with (ROOT / "gpu3.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        def occupied():
            probe = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=15, check=True)
            return [line.strip() for line in probe.stdout.splitlines() if env["CUDA_VISIBLE_DEVICES"] in line]
        for task in args.tasks:
            for case in args.cases:
                variants = [("eager", "none"), ("compile-default", "none")]
                if args.profiles:
                    variants += [("compile-default", "torch"), ("compile-default", "nsys")]
                if args.nsys_only:
                    variants = [("compile-default", "nsys")]
                for variant, profile in variants:
                    name = f"{task}-{case}-{variant}-{profile}"
                    target = out / (name + ".json")
                    cmd = [sys.executable, "-m", "wanbench", "smoke", "--task", task,
                           "--case", case, "--variant", variant, "--profile", profile,
                           "--samples", "50", "--warmup", "10", "--output", str(target)]
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
                           "returncode": rc, "artifact": str(target) if target.exists() else None}
                    if target.exists():
                        data = load_json(target)
                        row.update(status=data["status"], p50_ms=data.get("p50_ms"),
                                   max_errors=[x.get("max_abs_error") for x in data["correctness"]["checks"]])
                    if profile == "nsys":
                        row["nsys_report_exists"] = (out / (name + ".nsys-rep")).is_file()
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
    plan["status"] = "complete" if all(r["returncode"] == 0 for r in plan["results"]) else "completed-with-failures"
    dump_new(out / "summary.json", plan)
    return 0 if plan["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
