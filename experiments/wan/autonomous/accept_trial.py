"""Post-run independent acceptance; never feeds results back to the autonomous worker."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
GPU = 'GPU-f1b73c7a-e7a3-dd24-a71a-a8f7ebe1f3e8'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trial', type=Path, required=True)
    ap.add_argument('--reviewed-code', action='store_true', required=True)
    args = ap.parse_args()
    trial = args.trial.resolve()
    record = json.loads((trial / 'trial.json').read_text())
    if record['status'] != 'finished' or record['protected_changed']:
        raise ValueError('trial must be finished with protected files unchanged')
    if digest(trial / 'workspace/kernel.py') != record['candidate_sha256']:
        raise ValueError('candidate changed after trial')
    out = trial / 'acceptance'
    out.mkdir(exist_ok=False)
    snapshot = out / 'candidate'
    snapshot.mkdir()
    shutil.copy2(trial / 'workspace/kernel.py', snapshot / 'kernel.py')
    shutil.copy2(Path(__file__).with_name('external_adapter.py'), snapshot / 'adapter.py')
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=GPU, PYTHONUNBUFFERED='1',
               PYTHONPATH=str(ROOT) + ':' + str(trial / 'workspace'),
               WANBENCH_NSYS_CAPTURE='none', WANBENCH_NSYS_TRACE='cuda-sw')
    results = []
    with (ROOT / 'gpu3.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        def occupied():
            p = subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
            return [line for line in p.stdout.splitlines() if GPU in line]
        jobs=[(case,variant,'none') for case in ('debug','medium','long') for variant in ('compile-default','candidate')]
        jobs += [('debug',variant,profile) for profile in ('torch','nsys') for variant in ('compile-default','candidate')]
        for case,variant,profile in jobs:
            if occupied():
                raise RuntimeError('GPU busy; leave processes untouched')
            name=f'{case}-{variant}-{profile}'
            target=out/(name+'.json')
            command=[sys.executable,'-m','wanbench','smoke','--task','wan-rmsnorm','--case',case,
                     '--variant',variant,'--profile',profile,'--seed','607','--warmup','10','--samples','50',
                     '--timing','cuda-graph' if profile=='none' else 'single-call','--output',str(target)]
            if variant=='candidate':
                command += ['--candidate',str(snapshot/'adapter.py')]
            if profile=='nsys':
                command=['nsys','profile','--trace=cuda-sw,nvtx,osrt','--sample=none','--capture-range=none','--output='+str(out/name)]+command
            with (out/(name+'.log')).open('w') as log:
                p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    rc=p.wait(timeout=420)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGKILL); p.wait(); rc=124
            data=json.loads(target.read_text()) if target.exists() else {}
            row=dict(case=case,variant=variant,profile=profile,returncode=rc,status=data.get('status','missing'),p50_ms=data.get('p50_ms'),file=target.name)
            results.append(row)
            (out/'summary.json').write_text(json.dumps(dict(status='running',seed=607,candidate_sha256=record['candidate_sha256'],results=results),indent=2))
            print(json.dumps(row),flush=True)
            if rc==124 or occupied():
                raise RuntimeError('timeout or GPU still busy; acceptance stopped')
        (out/'summary.json').write_text(json.dumps(dict(status='finished',seed=607,candidate_sha256=record['candidate_sha256'],results=results),indent=2))


if __name__=='__main__':
    main()
