"""One bounded generation step of the AutoKernel Wan Phase B adapter.

Generation and execution are separated for source review. This does not benchmark
or automatically trust a generated candidate. No pre-written optimized answer is seeded.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import time
import tomllib

from wanbench.core import digest, dump_new
from wanbench.tasks import TASK_IDS

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=TASK_IDS)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--seconds', type=int, default=240)
    ap.add_argument('--previous', type=Path)
    ap.add_argument('--feedback', type=Path)
    args = ap.parse_args()
    if args.seconds <= 0 or args.seconds > 900:
        raise ValueError('generation budget must be 1..900 seconds')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    schema = {'type': 'object', 'properties': {
        'hypothesis': {'type': 'string'}, 'source': {'type': 'string'},
        'limitations': {'type': 'string'}},
        'required': ['hypothesis', 'source', 'limitations'], 'additionalProperties': False}
    dump_new(out / 'response.schema.json', schema)
    upstream = (ROOT.parents[1] / 'program.md').read_text()
    prompt = f'''Implement one candidate for task {args.task} in the AutoKernel Wan fixed-task Phase B adapter.
The user has approved six fixed Wan operators on B300. This call performs ONLY the hypothesis and code-generation
step. External review, fixed benchmarking and KEEP/REVERT decisions follow this call. Do not run commands or tools,
write files, benchmark, or claim measured performance. Return the requested structured response with complete Python
source exposing run(*inputs). Implement a new Triton or CUDA kernel; do not merely return torch.compile(reference).
Preserve input shape, dtype, device, intermediate BF16 rounding, and the reference function's exact public interface.
Support the three public sequence grids (256, 7800, 32760 tokens); do not specialize on test seed or tensor contents.
Use only torch, triton, triton.language and Python standard math. No IO, subprocesses, environment reads, hooks,
monkeypatching, custom streams, cached answers, tests introspection or timing manipulation. Use the current CUDA stream.
Input-dependent work must occur inside run. Pure launch metadata may be prepared in Python. Grid sizes are CPU metadata.
Prefer one focused optimization. Avoid unbounded loops. Leave all inputs unchanged.
Hardware: NVIDIA B300 capability 10.3; Python 3.12; NVIDIA torch 2.12.0a0 CUDA 13.2; Triton 3.6.0.
This is exploratory synthetic bring-up, not a formal framework score or whole-model experiment.
The following upstream program is methodological context. Its Phase A/C, file reads, commands, never-stop rule and
benchmark paths are superseded by the bounded generation-only contract above. Retain its optimization reasoning.

UPSTREAM PROGRAM:\n{upstream}

FROZEN REFERENCE AND FIXTURES:\n{(ROOT / 'wanbench/tasks.py').read_text()}
'''
    if args.previous:
        prompt += '\nCURRENT BEST SOURCE:\n' + args.previous.read_text()
    if args.feedback:
        prompt += '\nMEASURED FEEDBACK FROM PREVIOUS STEP:\n' + args.feedback.read_text()
    prompt_path = out / 'prompt.txt'
    prompt_path.write_text(prompt)
    # The prompt is passed on stdin, never interpolated into a shell command.
    command = ['codex', 'exec', '-s', 'read-only', '--ephemeral', '--json',
               '--output-schema', str(out / 'response.schema.json'),
               '-o', str(out / 'response.json'), '-']
    started = time.monotonic()
    with prompt_path.open() as input_file, (out / 'agent-events.jsonl').open('w') as log:
        proc = subprocess.Popen(command, stdin=input_file, stdout=log, stderr=subprocess.STDOUT,
                                cwd=ROOT, start_new_session=True)
        try:
            rc = proc.wait(timeout=args.seconds)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(); rc = 124
    cfg = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "config.toml"
    public_config = {}
    if cfg.is_file():
        values = tomllib.loads(cfg.read_text())
        public_config = {key: values.get(key) for key in ["model", "model_reasoning_effort"]}
    record = {"configured_model": public_config,'framework': 'autokernel-wan-fixed-task-adapter', 'task': args.task,
              'kind': 'generation-step', 'formal_ready': False, 'returncode': rc,
              'wall_seconds': time.monotonic() - started, 'budget_seconds': args.seconds,
              'upstream_program_sha256': digest(ROOT.parents[1] / 'program.md'),
              'prompt_sha256': digest(prompt_path), 'status': 'generation-failed',
              'candidate_executed': False, 'token_usage': []}
    for line in (out / 'agent-events.jsonl').read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get('usage'):
            record['token_usage'].append(event['usage'])
    if rc == 0 and (out / 'response.json').is_file():
        response = json.loads((out / 'response.json').read_text())
        source = response['source']
        ast.parse(source)
        (out / 'kernel.py').write_text(source)
        record.update(status='awaiting-source-review', candidate_sha256=digest(out / 'kernel.py'),
                      hypothesis=response['hypothesis'], limitations=response['limitations'])
    dump_new(out / 'generation.json', record)
    print(json.dumps(record, indent=2))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
