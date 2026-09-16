"""Run precision diagnostics sequentially with the campaign GPU guard."""
import argparse
import datetime
import fcntl
import os
import subprocess
import sys
from pathlib import Path
from preflight import ROOT, run_bounded
from wanbench.core import dump_new, load_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cases', nargs='+', choices=['debug','medium','long'], default=['debug','medium','long'])
    ap.add_argument('--followup', action='store_true')
    ap.add_argument('--tasks', nargs='+', choices=['wan-linear-gelu','t2-qknorm-rope'], default=['wan-linear-gelu','t2-qknorm-rope'])
    args = ap.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=load_json(ROOT/'configs/device.json')['uuid'],
               PYTHONPATH=str(ROOT), PYTHONUNBUFFERED='1')
    env['TORCH_LOGS'] = 'output_code'
    env['TORCHINDUCTOR_CACHE_DIR'] = str(out/'inductor-cache')
    result = {'kind': 'precision-campaign', 'results': [], 'status': 'running'}
    def occupied():
        p = subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=15)
        return [s for s in p.stdout.splitlines() if env['CUDA_VISIBLE_DEVICES'] in s]
    with (ROOT/'gpu3.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for task in args.tasks:
            for case in args.cases:
                if occupied():
                    result['status']='blocked-gpu-busy'; dump_new(out/'summary.json',result); return 3
                target=out/f'{task}-{case}.json'
                start=datetime.datetime.now(datetime.timezone.utc).isoformat()
                rc=run_bounded([sys.executable,'precision_followup.py' if args.followup else 'precision_diagnose.py','--task',task,'--case',case,'--output',str(target)],out/f'{task}-{case}.log',env,420)
                row={'task':task,'case':case,'returncode':rc,'started':start,'artifact_exists':target.exists(),'occupancy_after':occupied()}
                result['results'].append(row)
                (out/'progress.json').write_text(__import__('json').dumps(result,indent=2))
                print(row,flush=True)
                if rc==124 or row['occupancy_after']:
                    result['status']='stopped-inspect-required'; dump_new(out/'summary.json',result); return 3
    result['status']='complete' if all(r['returncode']==0 and r['artifact_exists'] for r in result['results']) else 'completed-with-errors'
    dump_new(out/'summary.json',result)


if __name__ == '__main__':
    main()
