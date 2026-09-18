"""Package completed trial evidence for operator review and local backup, not worker feedback."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--trial',type=Path,required=True)
    args=ap.parse_args()
    trial=args.trial.resolve()
    record=json.loads((trial/'trial.json').read_text())
    if record['status']!='finished':
        raise ValueError('export only after worker completion')
    archive=trial/'evidence.tar.gz'
    chosen=[]
    for name in ['trial.json','prompt.txt','final.txt','agent-events.jsonl','agent-stderr.log','setup.log']:
        if (trial/name).is_file(): chosen.append(trial/name)
    work=trial/'workspace'
    for name in ['kernel.py','results.tsv','run.log']:
        if (work/name).is_file():chosen.append(work/name)
    for folder in [work/'workspace',work/'.git',trial/'acceptance']:
        if folder.exists():
            chosen.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.cache' not in p.parts)
    # Bound the transfer. Large traces stay on the experiment machine and are explicitly listed.
    omitted=[dict(path=str(p.relative_to(trial)),bytes=p.stat().st_size) for p in chosen if p.stat().st_size>4_000_000]
    chosen=[p for p in chosen if p.stat().st_size<=4_000_000]
    manifest=[dict(path=str(p.relative_to(trial)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(chosen))]
    (trial/'evidence-manifest.json').write_text(json.dumps(dict(files=manifest,omitted=omitted),indent=2))
    with tarfile.open(archive,'w:gz') as dest:
        for p in sorted(set(chosen)):
            dest.add(p,arcname=str(p.relative_to(trial)))
        dest.add(trial/'evidence-manifest.json',arcname='evidence-manifest.json')
    data=archive.read_bytes()
    print(json.dumps(dict(file=str(archive),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),file_count=len(manifest),omitted=omitted),indent=2))


if __name__=='__main__':main()
