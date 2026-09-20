"""External recorder for an unchanged AutoKernel program.md pilot."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tarfile
import time
import tomllib

UPSTREAM = "78435821cc3d5756ba6ee1785c397f6d8fa8c90d"


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def command(args, cwd=None, env=None):
    return subprocess.check_output(args, cwd=cwd, env=env, text=True)


def main():
    ap = argparse.ArgumentParser()
    for name in ("trial", "repo", "venv"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--gpu", required=True)
    args = ap.parse_args()
    trial = args.trial.resolve()
    trial.mkdir(parents=True, exist_ok=False)
    work = trial / "workspace"
    work.mkdir()
    env = dict(os.environ)
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID"):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1", UV_NO_SYNC="1",
               VIRTUAL_ENV=str(args.venv), UV_PROJECT_ENVIRONMENT=str(args.venv),
               PATH=str(args.venv / "bin") + ":" + env["PATH"],
               XDG_CACHE_HOME=str(work / ".cache"), TMPDIR=str(work / ".tmp"))
    for folder in (work / ".cache", work / ".tmp"):
        folder.mkdir()
    with (trial.parent / (args.gpu + ".lock")).open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        occupancy = command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"])
        memory = command(["nvidia-smi", "--id=" + args.gpu,
                          "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        if args.gpu in occupancy or int(memory.strip()) > 100:
            raise RuntimeError("Selected GPU is occupied; no Agent started")
        archive = trial / "upstream.tar"
        subprocess.run(["git", "-C", str(args.repo), "archive", "-o", str(archive), UPSTREAM], check=True)
        with tarfile.open(archive) as source:
            source.extractall(work, filter="data")
        names = command(["git", "-C", str(args.repo), "ls-tree", "-r", "--name-only", UPSTREAM]).splitlines()
        protected = {name: digest(work / name) for name in names if name not in {"kernel.py", "progress.png"}}
        for cmd in (["git", "init", "-b", "codex/official-h200-agent-r01"],
                    ["git", "config", "user.name", "AutoKernel Trial"],
                    ["git", "config", "user.email", "autokernel-trial@localhost"],
                    ["git", "config", "gc.auto", "0"], ["git", "add", "."],
                    ["git", "commit", "-m", "Frozen unmodified AutoKernel upstream"]):
            command(cmd, work)
        cfg = Path(env.get("CODEX_HOME", str(Path.home() / ".codex"))) / "config.toml"
        config = tomllib.loads(cfg.read_text()) if cfg.exists() else {}
        started_at = dt.datetime.now(dt.timezone.utc)
        wrapup = started_at + dt.timedelta(seconds=1080)
        prompt = f"""Read program.md and run a new AutoKernel experiment from setup through integration.
Task: the repository's models/llama_7b.py, class LlamaModel, input shape 1,512, dtype float16. Use the native top-5 extraction and priority workflow.
The human has approved this model, the native optimization plan, and a 1200-second pilot budget. Proceed autonomously without waiting for confirmation. Use the existing codex/ branch or another codex/ branch.
Follow the native hypothesis/edit/commit/benchmark/keep-or-revert loop. Choose your own hypotheses, tools and experiments. No operator advice will arrive. Do not spawn other agents.
The Python environment is already installed. Use uv run with the provided UV_PROJECT_ENVIRONMENT and UV_NO_SYNC=1. Do not install or replace dependencies. Only the assigned GPU is visible; never change CUDA_VISIBLE_DEVICES or use another GPU.
Do not modify any framework implementation, model, reference, benchmark, numerical tolerance, or workflow instruction. You may modify candidate kernel.py, create/save candidates and diagnostics under workspace/, and write native logs, notes, results.tsv and generated reports/plots. Do not change files under kernels/; use those unmodified starters as instructed by program.md.
Do not inspect parent or sibling project directories, previous experiments, personal memory, other optimized candidates, or credentials. The specified Python environment and installed library source may be read as needed. Do not use a prior solution or ask the coordinating session to repair code.
All input-dependent computation must remain inside the candidate call. Do not cache answers, inspect tests to select answers, mutate inputs, or change timing or validation code.
The hard budget is 1200 seconds from launch. Begin wrapping up by {wrapup.isoformat()} and attempt the native integration and final report within the remaining budget. Budget takes precedence over the native never-stop wording; do not claim untested kernels are optimized.
Preserve each experiment's hypothesis, logs, measurements and keep/revert decision. Keep the best valid candidates in the native locations. Do not publish or push. Finish by reporting actual results, limitations, remaining work and your stopping reason. If blocked, preserve the state and explain the blocker without changing the evaluation contract.
"""
        (trial / "prompt.txt").write_text(prompt)
        record = dict(status="running", trial=trial.name, upstream=UPSTREAM, gpu_uuid=args.gpu,
                      model=config.get("model"), reasoning_effort=config.get("model_reasoning_effort"),
                      cli_version=command(["codex", "--version"], env=env).strip(),
                      started_at=started_at.isoformat(), hard_budget_seconds=1200,
                      protected_before=protected, prompt_sha256=digest(trial / "prompt.txt"),
                      operator_interventions=[], framework_mode="native-program-full-model-pilot",
                      host_overrides={"hooks": False, "context-mode": False,
                                      "sandbox": "danger-full-access", "approval_policy": "never"},
                      isolation="container and contractual boundaries; not a strong security sandbox")
        save = lambda: (trial / "trial.json").write_text(json.dumps(record, indent=2))
        save()
        started = time.monotonic()
        with (trial / "prompt.txt").open() as inp, (trial / "agent-events.jsonl").open("w") as out, (trial / "agent-stderr.log").open("w") as err:
            proc = subprocess.Popen(["codex", "exec", "--ephemeral", "--json", "-s", "danger-full-access",
                                     "-c", 'approval_policy="never"', "-c", "mcp_servers.context-mode.enabled=false",
                                     "--disable", "hooks", "-o", str(trial / "final.txt"), "-"],
                                    cwd=work, env=env, stdin=inp, stdout=out, stderr=err, start_new_session=True)
            record["pid"] = proc.pid
            save()
            try:
                rc = proc.wait(timeout=1200)
                stop = "agent-exited"
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                rc, stop = 124, "external-budget-stop"
        record.update(status="finished", returncode=rc, stop_reason=stop,
                      wall_seconds=time.monotonic() - started,
                      finished_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                      protected_changed=[n for n,h in protected.items() if not (work/n).is_file() or digest(work/n)!=h],
                      candidate_sha256=digest(work / "kernel.py") if (work / "kernel.py").is_file() else None)
        record["token_usage"] = []
        for line in (trial / "agent-events.jsonl").read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("usage"):
                record["token_usage"].append(event["usage"])
        save()
        def keep(info):
            return None if any(x in {".cache", ".tmp", ".venv", "__pycache__"} for x in Path(info.name).parts) else info
        with tarfile.open(trial / "workspace-final.tar.gz", "w:gz") as out:
            out.add(work, arcname="workspace", filter=keep)
        record["workspace_archive_sha256"] = digest(trial / "workspace-final.tar.gz")
        save()
        print(json.dumps({k:v for k,v in record.items() if k != "protected_before"}, indent=2))


if __name__ == "__main__":
    main()
