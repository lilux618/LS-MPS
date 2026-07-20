#!/usr/bin/env python3
"""Run the deterministic smoke case twice and compare canonical CSV/JSON bytes."""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path
FILES=['metrics.csv','sampling_windows.csv','flow_sections.csv','mass_conservation.csv','summary.json']
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='outputs/reproducibility'); ap.add_argument('--steps',type=int,default=36); a=ap.parse_args()
    root=Path(a.out); shutil.rmtree(root,ignore_errors=True); (root/'run_a').mkdir(parents=True); (root/'run_b').mkdir(parents=True)
    for name in ['run_a','run_b']:
        subprocess.run([sys.executable,'python/rain_plate_cpu.py','--output',str(root/name),'--steps',str(a.steps),'--validation-config','config/rain_plate_validation.json'],check=True)
    mismatch=[]; hashes={}
    for f in FILES:
        ha=digest(root/'run_a'/f); hb=digest(root/'run_b'/f); hashes[f]={'run_a':ha,'run_b':hb,'match':ha==hb}
        if ha!=hb: mismatch.append(f)
    result={'passed':not mismatch,'mismatched_files':mismatch,'hashes':hashes}; (root/'reproducibility.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2)); raise SystemExit(0 if result['passed'] else 2)
if __name__=='__main__': main()
