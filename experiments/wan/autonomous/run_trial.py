"""Launch one native AutoKernel custom-problem trial, without repair or feedback injection."""
import argparse
import datetime
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

UPSTREAM = '78435821cc3d5756ba6ee1785c397f6d8fa8c90d'
GPU = 'GPU-f1b73c7a-e7a3-dd24-a71a-a8f7ebe1f3e8'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, cwd=None, env=None):
    return subprocess.run(command, cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trial', required=True, type=Path)
    ap.add_argument('--venv', required=True, type=Path)
    args = ap.parse_args()
    trial = args.trial.resolve()
    trial.mkdir(parents=True, exist_ok=False)
    work = trial / 'workspace'
    work.mkdir()
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=GPU, PYTHONUNBUFFERED='1',
               VIRTUAL_ENV=str(args.venv), UV_PROJECT_ENVIRONMENT=str(args.venv), UV_NO_SYNC='1',
               XDG_CACHE_HOME=str(work / '.cache'), TMPDIR=str(work / '.tmp'),
               PATH=str(args.venv / 'bin') + ':' + env['PATH'])
    for directory in (work / '.cache', work / '.tmp'):
        directory.mkdir()
    with (ROOT / 'gpu3.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        occupancy = run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'])
        if GPU in occupancy:
            raise RuntimeError('selected GPU is busy; trial not started')
        archive = trial / 'upstream.tar'
        subprocess.run(['git', '-C', str(ROOT.parents[1]), 'archive', '--format=tar', '-o', str(archive), UPSTREAM], check=True)
        with tarfile.open(archive) as source:
            source.extractall(work, filter='data')
        (work / 'task.py').write_bytes((HERE / 'wan_rmsnorm_problem.py').read_bytes())
        setup = run([str(args.venv / 'bin/python'), 'kernelbench/bridge.py', 'setup',
                     '--level', '1', '--problem', '9001', '--source', 'file',
                     '--file-path', 'task.py', '--backend', 'triton'], work, env)
        (trial / 'setup.log').write_text(setup)
        run(['git', 'init', '-b', 'codex/native-rmsnorm-r01'], work)
        run(['git', 'config', 'user.name', 'AutoKernel Trial'], work)
        run(['git', 'config', 'user.email', 'autokernel-trial@localhost'], work)
        run(['git', 'add', '.'], work)
        run(['git', 'commit', '-m', 'Frozen upstream and reference-only Wan task'], work)
        protected = {str(p.relative_to(work)): sha(p) for p in work.rglob('*')
                     if p.is_file() and '.git' not in p.parts and '.cache' not in p.parts
                     and '.tmp' not in p.parts and '__pycache__' not in p.parts and p.name != 'kernel.py'}
        deadline = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=810)
        prompt = f'''Run the native AutoKernel autonomous optimization workflow in kernelbench/program_kb.md.
The custom-file problem is already set up by the native bridge. It is Wan RMSNorm, not an official KernelBench dataset entry.
Read the native instructions and task/reference. Optimize independently using the native hypothesis/edit/commit/benchmark/keep-or-revert loop.
You may use the shell, compiler, profiler, installed package source and diagnostic tools. No further operator advice will arrive.
The user approved this experiment and its tools. Do not wait for confirmation. Do not spawn other agents.
Use the existing environment: uv run commands have UV_NO_SYNC=1 and the supplied environment. Do not install or replace dependencies.
Only the assigned GPU is visible. Never change CUDA_VISIBLE_DEVICES or use another GPU.
Task contract: support contiguous BF16 x of [1,S,1536] and BF16 weight [1536], S in [256,7800,32760], following the frozen reference.
The native benchmark uses S=7800. Independent final acceptance will also check other declared lengths and new random inputs, unchanged inputs, dtype and device.
The native benchmark reports eager-relative timing. Final external performance compares native torch.compile default; do not mislabel the internal speedup.
Do not alter task.py, workspace/kb_active/reference.py, benchmark code, workflow instructions or other framework implementation.
Only candidate kernel.py and new diagnostic scripts/logs/notes may be changed. Keep reference helpers/get_inputs in kernel.py unchanged.
Do not inspect parent/sibling project directories, previous trial results, prior experiment reports, or existing optimized Wan candidates. Read installed libraries only for source diagnosis.
All input-dependent computation must remain inside forward; no cached answers, seed-specific logic, timing changes, test introspection or input mutation.
The upstream starter is allowed; generate your own optimization. Do not use an external answer or ask the coordinating session for a repair.
The trial has a 900-second hard wall-clock budget. Finish by {deadline.isoformat()} (810 seconds after launch allowance) to leave time for preservation.
Budget takes precedence over the upstream never-stop wording. You decide the number of trials and when to stop within budget.
Save each experiment's hypothesis, measurement and keep/revert decision in results.tsv; preserve diagnostic logs under workspace/.
Do not publish or push. Finish with the best candidate in kernel.py and a final explanation citing actual tool results, limitations and failures.
If blocked, report the blocker and preserve the state; do not change the evaluation contract to obtain a pass.
'''
        (trial / 'prompt.txt').write_text(prompt)
        cfg = Path(os.environ.get('CODEX_HOME', str(Path.home() / '.codex'))) / 'config.toml'
        values = tomllib.loads(cfg.read_text()) if cfg.exists() else {}
        record = dict(trial=trial.name, status='running', upstream=UPSTREAM, gpu_uuid=GPU,
                      model=values.get('model'), reasoning_effort=values.get('model_reasoning_effort'),
                      framework_mode='native-kernelbench-custom-file', hard_budget_seconds=900,
                      prompt_sha256=sha(trial / 'prompt.txt'), protected_before=protected,
                      started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      operator_interventions=[], formal_ready=False)
        (trial / 'trial.json').write_text(json.dumps(record, indent=2))
        started = time.monotonic()
        with (trial / 'prompt.txt').open() as inp, (trial / 'agent-events.jsonl').open('w') as out, (trial / 'agent-stderr.log').open('w') as err:
            proc = subprocess.Popen(['codex', 'exec', '-s', 'workspace-write', '-c', 'approval_policy="never"',
                                     '--ephemeral', '--json', '-o', str(trial / 'final.txt'), '-'],
                                    cwd=work, env=env, stdin=inp, stdout=out, stderr=err, start_new_session=True)
            record['pid'] = proc.pid
            (trial / 'trial.json').write_text(json.dumps(record, indent=2))
            try:
                rc = proc.wait(timeout=900)
                stop = 'agent-exited'
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                rc, stop = 124, 'external-budget-stop'
        record.update(status='finished', returncode=rc, stop_reason=stop, wall_seconds=time.monotonic()-started,
                      finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      candidate_sha256=sha(work / 'kernel.py'),
                      protected_changed=[name for name,h in protected.items() if not (work/name).exists() or sha(work/name)!=h])
        usages=[]
        counts={}
        for line in (trial / 'agent-events.jsonl').read_text().splitlines():
            try:
                event=json.loads(line)
            except ValueError:
                continue
            if event.get('usage'):
                usages.append(event['usage'])
            item=event.get('item',{})
            if event.get('type')=='item.completed':
                kind=item.get('type','unknown'); counts[kind]=counts.get(kind,0)+1
        record.update(token_usage=usages, completed_item_counts=counts,
                      occupancy_after=run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader']))
        # Preserve the final state and git objects, including rollback history, before independent acceptance.
        with tarfile.open(trial / 'workspace-final.tar.gz','w:gz') as out:
            for item in work.iterdir():
                if item.name not in {'.cache','.tmp','.venv'}:
                    out.add(item,arcname=item.name)
        (trial / 'trial.json').write_text(json.dumps(record,indent=2))
        print(json.dumps({k:v for k,v in record.items() if k not in {'protected_before'}},indent=2))


if __name__ == '__main__':
    main()
