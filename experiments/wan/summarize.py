"""Summarize actual smoke artifacts; profiled times never enter speed ratios."""
import argparse
from collections import Counter
import csv
import io
import json
import os
from pathlib import Path
import subprocess

from wanbench.core import digest, dump_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directories', type=Path, nargs='+')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--nsys-stats', action='store_true')
    args = ap.parse_args()
    rows = []
    for directory in args.directories:
        for path in sorted(directory.glob('*.json')):
            if path.name.endswith('.trace.json') or path.name in {'summary.json', 'progress.json'}:
                continue
            data = json.loads(path.read_text())
            if data.get('kind') != 'exploratory-gpu':
                continue
            row = {k: data.get(k) for k in ['task', 'case', 'variant', 'profile', 'status', 'p50_ms']}
            row['measurement'] = data.get('measurement')
            row['correctness_checks'] = data.get('correctness', {}).get('checks', [])
            row['graph_correctness'] = data.get('graph_correctness')
            if data.get('samples_ms'):
                ordered = sorted(data['samples_ms'])
                row['p10_ms'] = ordered[int((len(ordered) - 1) * .1)]
                row['p90_ms'] = ordered[int((len(ordered) - 1) * .9)]
            row['artifact'] = str(path)
            row['sha256'] = digest(path)
            if data['profile'] == 'torch' and data.get('trace'):
                trace = Path(data['trace'])
                events = json.loads(trace.read_text()).get('traceEvents', [])
                kernels = [e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X']
                row['profiled_calls'] = 5
                row['kernel_count'] = len(kernels)
                row['kernel_duration_sum_us'] = sum(e.get('dur', 0) for e in kernels)
                row['kernel_names'] = dict(Counter(e['name'] for e in kernels))
            if data['profile'] == 'nsys':
                row['capture_mode'] = data.get('nsys_capture_mode', 'cudaProfilerApi')
                report = path.with_suffix('.nsys-rep')
                row['report_exists'] = report.is_file()
                if report.is_file() and args.nsys_stats:
                    # NVIDIA Nsight Systems UserGuide, CLI nsys stats examples.
                    target = path.with_suffix('.stats.log')
                    env = dict(os.environ, TMPDIR=str(directory.resolve() / 'tmp'))
                    proc = subprocess.run(['nsys', 'stats', '--report', 'cuda_gpu_kern_sum',
                                           '--format', 'csv', str(report)], capture_output=True,
                                          text=True, env=env, timeout=120)
                    target.write_text(proc.stdout + proc.stderr)
                    row['stats_returncode'] = proc.returncode
                    lines = proc.stdout.splitlines()
                    start = next((i for i, line in enumerate(lines) if 'Total Time (ns)' in line and 'Instances' in line), None)
                    if start is not None:
                        entries = list(csv.DictReader(io.StringIO('\n'.join(lines[start:]))))
                        entries = [r for r in entries if (r.get('Instances') or '').isdigit()]
                        row['profiled_calls'] = 5 if row['capture_mode'] == 'cudaProfilerApi' else None
                        row['scope'] = 'measurement calls' if row['capture_mode'] == 'cudaProfilerApi' else 'entire process, diagnostic only'
                        row['kernel_count'] = sum(int(r['Instances']) for r in entries)
                        row['kernel_duration_sum_us'] = sum(float(r['Total Time (ns)']) for r in entries) / 1000
                        row['kernel_names'] = {r['Name']: int(r['Instances']) for r in entries}
            rows.append(row)
    dump_new(args.output, {'kind': 'synthetic-preflight-summary', 'formal_ready': False, 'results': rows})
    for r in rows:
        us = round(r['p50_ms'] * 1000, 3) if r.get('p50_ms') is not None else '-'
        print(r['task'], r['case'], r['variant'], r['profile'], r['status'], us, 'us',
              'kernels=', r.get('kernel_count', '-'))


if __name__ == '__main__':
    main()
